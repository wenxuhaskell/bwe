import argparse
import tkinter
from tkinter import filedialog, Tk, Button, Label, messagebox
from typing import Any, Dict, List
import d3rlpy
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from sklearn.preprocessing import MinMaxScaler
from d3rlpy.models.encoders import register_encoder_factory

from BweEncoder import LSTMEncoderFactory, ACEncoderFactory
from BweUtils import load_test_data, load_train_data_from_file
from BweReward import Feature, MI, MIType, get_feature, get_feature_for_mi, get_decay_weights, reward_bwe

model_filename = ''
data_filenames =[]

# short-term MI
def smi_features(feature_vec: List[float]) -> float:
    # use the 5 recent short MIs. (TODO: refinement)
    receive_rate = np.sum(feature_vec[(Feature.RECV_RATE - 1) * 10 : (Feature.RECV_RATE - 1) * 10 + 5]) / 5
    queuing_delay = np.sum(feature_vec[(Feature.QUEUING_DELAY - 1) * 10 : (Feature.QUEUING_DELAY - 1) * 10 + 5]) / 5
    pkt_loss_rate = np.sum(feature_vec[(Feature.PKT_LOSS_RATIO - 1) * 10 : (Feature.PKT_LOSS_RATIO - 1) * 10 + 5]) / 5

    return 0.6 * np.log(4 * receive_rate + 1) - queuing_delay / 1000 - 10 * pkt_loss_rate

# long-term MI
def lmi_features(feature_vec: List[float]) -> float:
    # use the 5 recent short MIs. (TODO: refinement)
    receive_rate = np.sum(feature_vec[(Feature.RECV_RATE - 1) * 10 : (Feature.RECV_RATE - 1) * 10 + 5]) / 5
    queuing_delay = np.sum(feature_vec[(Feature.QUEUING_DELAY - 1) * 10 : (Feature.QUEUING_DELAY - 1) * 10 + 5]) / 5
    pkt_loss_rate = np.sum(feature_vec[(Feature.PKT_LOSS_RATIO - 1) * 10 : (Feature.PKT_LOSS_RATIO - 1) * 10 + 5]) / 5

    return 0.6 * np.log(4 * receive_rate + 1) - queuing_delay / 1000 - 10 * pkt_loss_rate

def rewards_qoe(observation: List[float], rf_params: Dict[str, Any]) -> float:
    qoes = dict(zip(MI, [0.0] * len(MI)))
    # 1. rate QoE: 0...1
    qoe_rates = dict(zip(MI, [0.0] * len(MI)))
    rates = get_feature_for_mi(observation, "RECV_RATE", MIType.ALL)
    for i, rate in enumerate(rates):
        rf_params["MAX_RATE"][MI(i + 1)] = max(rf_params["MAX_RATE"][MI(i + 1)], rate)
        max_rate = rf_params["MAX_RATE"][MI(i + 1)] if rf_params["MAX_RATE"][MI(i + 1)] > 0 else 1
        # logarithmic nature scaled by the maximum observed rate in this MI
        qoe_rates[MI(i + 1)] = np.log((np.exp(1) - 1) * (rate / max_rate) + 1)

    # 2. delay QoE: 0...1
    qoe_delays = dict(zip(MI, [0.0] * len(MI)))
    delays = get_feature_for_mi(observation, "DELAY", MIType.ALL)
    min_seen_delays = get_feature_for_mi(observation, "MIN_SEEN_DELAY", MIType.ALL)
    for i, delay in enumerate(delays):
        delay += 200  # add a substracted base delay of 200 ms to have an absolute value
        # d_max - d / d_max - d_min
        rf_params["MAX_DELAY"][MI(i + 1)] = max(rf_params["MAX_DELAY"][MI(i + 1)], delay)
        max_delay = rf_params["MAX_DELAY"][MI(i + 1)] if rf_params["MAX_DELAY"][MI(i + 1)] > 0 else delay
        qoe_delays[MI(i + 1)] = (
            (max_delay - delay) / (max_delay - min_seen_delays[i]) if max_delay > min_seen_delays[i] else 0
        )

    # 3. loss QoE: 0...1
    qoe_losses = dict(zip(MI, [0.0] * len(MI)))
    losses = get_feature_for_mi(observation, "PKT_LOSS_RATIO", MIType.ALL)
    for i, loss in enumerate(losses):
        qoe_losses[MI(i + 1)] = 1 - loss

    # 4. jitter QoE: 0...1
    qoe_jitters = dict(zip(MI, [0.0] * len(MI)))
    jitters = get_feature_for_mi(observation, "PKT_JITTER", MIType.ALL)
    for i, jitter in enumerate(jitters):
        qoe_jitters[MI(i + 1)] = -0.04 * np.sqrt(min(625, jitter)) + 1

    # combine all QoEs
    # FIXME: tune the weights!
    for mi in MI:
        qoes[mi] = 0.3 * qoe_rates[mi] + 0.3 * qoe_delays[mi] + 0.3 * qoe_losses[mi] + 0.1 * qoe_jitters[mi]

    # QoE long term is more important than short term, 0.66 vs 0.33 SO FAR,
    # FIXME: tune the weights!
    short_qoe = 0.0
    long_qoe = 0.0
    mi_weights = get_decay_weights(5)
    mi_weights = np.concatenate((mi_weights, mi_weights))
    for mi in MI:
        w = mi_weights[mi - 1]
        if mi <= MI.SHORT_300:
            short_qoe += w * qoes[mi]
        else:
            long_qoe += w * qoes[mi]

    # final QoE: 0..5
    final_qoe = 0.0
    final_qoe = 0.33 * short_qoe + 0.66 * long_qoe
    final_qoe *= 5
    return final_qoe, (short_qoe*5), (long_qoe*5)


def rewards_bwe(observation: List[float], rf_params: Dict[str, Any]=None) -> float:
    # use the 5 recent short MIs.
    # to adward
    video_pkt_prob = np.sum(observation[(Feature.VIDEO_PKT_PROB - 1) * 10: (Feature.VIDEO_PKT_PROB - 1) * 10 + 5]) / 5
    receive_rate = np.sum(observation[(Feature.RECV_RATE - 1) * 10 : (Feature.RECV_RATE - 1) * 10 + 5]) / 5

    award = (0.5*video_pkt_prob + 0.5*receive_rate)

    # to punish
    audio_pkt_prob = np.sum(observation[(Feature.AUDIO_PKT_PROB - 1) * 10 : (Feature.AUDIO_PKT_PROB - 1) * 10 + 5]) / 5
    pkt_interarrival = np.sum(observation[(Feature.PKT_INTERARRIVAL - 1) * 10 : (Feature.PKT_INTERARRIVAL - 1) * 10 + 5]) / 5
    pkt_jitter = np.sum(observation[(Feature.PKT_JITTER - 1) * 10 : (Feature.PKT_JITTER - 1) * 10 + 5]) / 5
    pkt_loss_rate = np.sum(observation[(Feature.PKT_LOSS_RATIO - 1) * 10: (Feature.PKT_LOSS_RATIO - 1) * 10 + 5]) / 5
    queuing_delay = np.sum(observation[(Feature.QUEUING_DELAY - 1) * 10 : (Feature.QUEUING_DELAY - 1) * 10 + 5]) / 5

    fine = (0.35*audio_pkt_prob + 0.35*pkt_interarrival + 0.1*pkt_jitter + 0.1*pkt_loss_rate + 0.1*queuing_delay)

    return (award - 0.8*fine + 0.2)*5


def get_feature_by_index(
    observation: List[float],
    feature_index: Feature,
    mi: MI = MI.LONG_600,
) -> float:
    return observation[(feature_index - 1) * 10 + mi - 1]


class eval_model:
    def __init__(self, initdir: str):
        self.__initdir = initdir
        self.__data_filenames = []
        self.__model_filename = ''
        self.__plot_log = False

    def run(self):
        window = Tk()

        window.title("Evaluator")

        window.geometry("600x400")

        window.config(background="grey")

        label_model = Label(text="Model file: " + self.__model_filename)

        self.select_model_file()

        label_model = Label(text="Model file: " + self.__model_filename)

        self.select_data_file()

        data_file_names = ''.join([o + '\n' for o in self.__data_filenames])
        label_data = Label(text=data_file_names)

        self.ask_plot_log()

        self.eval_model()

        label_model.grid(column=1, row=1)
        label_data.grid(column=1, row=2)
        window.mainloop()

    def select_model_file(self):
        self.__model_filename = filedialog.askopenfilename(initialdir=self.__initdir,
                                              title="Select a model file",
                                              filetypes=(("Model file", "*.d3"), ("All files", "*.*")))

    def select_data_file(self):
        self.__data_filenames = filedialog.askopenfilenames(initialdir=self.__initdir,
                                                title="Select data files",
                                                filetypes=(("Json files", "*.json"), ("Npz files", "*.npz")))

    def ask_plot_log(self):
        self.__plot_log = messagebox.askyesno("Plot log file?")

    def eval_model(self):
        if self.__model_filename == '':
            print("Please select the model file!")
            return

        if self.__data_filenames == []:
            print("Please select data files!")
            return

        print(f"Load the pre-trained model from the file {self.__model_filename}")

        # register your own encoder factory
        register_encoder_factory(ACEncoderFactory)
        register_encoder_factory(LSTMEncoderFactory)

        algo = d3rlpy.load_learnable(self.__model_filename, device='cuda:0' if torch.cuda.is_available() else 'cpu:0')
        # load the list of log files under the given directory
        # iterate over files in that directory

        predictions = []
        bw_predictions = []
        f_rwds = []
        s_rwds = []
        l_rwds = []
        inner_params = {
            "MAX_RATE": dict(zip(MI, [0.0] * len(MI))),
            "MAX_DELAY": dict(zip(MI, [0.0] * len(MI))),
        }

        # short mi features
        smi_f1 = []
        smi_f2 = []
        smi_f3 = []
        smi_f4 = []
        smi_f5 = []
        smi_f6 = []
        smi_f7 = []
        smi_f8 = []
        smi_f9 = []
        smi_f10 = []
        smi_f11 = []
        smi_f12 = []
        smi_f13 = []
        smi_f14 = []
        smi_f15 = []


        for filename in self.__data_filenames:
            result = load_train_data_from_file(filename)
            observations, bw_preds, _, _ = result
            obsScaler = MinMaxScaler()
            observations = obsScaler.fit_transform(observations)
            bw_predictions.append(bw_preds)

            # returns greedy action
            for observation in observations:
                if self.__plot_log:
                    # extract rewards
                    #f_reward, s_reward, l_reward = rewards_qoe(observation, inner_params)
                    f_reward = reward_bwe(observation)
                    f_rwds.append(f_reward)
                    #s_rwds.append(s_reward)
                    #l_rwds.append(l_reward)

                    # extract features for MI.SHORT_300
                    m_interval = MI.SHORT_300
                    smi_f1.append(get_feature_by_index(observation, Feature.RECV_RATE, m_interval))
                    smi_f2.append(get_feature_by_index(observation, Feature.RECV_PKT_AMOUNT, m_interval))
                    smi_f3.append(get_feature_by_index(observation, Feature.RECV_BYTES, m_interval))
                    smi_f4.append(get_feature_by_index(observation, Feature.QUEUING_DELAY, m_interval))
                    smi_f5.append(get_feature_by_index(observation, Feature.DELAY, m_interval))
                    smi_f6.append(get_feature_by_index(observation, Feature.MIN_SEEN_DELAY, m_interval))
                    smi_f7.append(get_feature_by_index(observation, Feature.DELAY_RATIO, m_interval))
                    smi_f8.append(get_feature_by_index(observation, Feature.DELAY_MIN_DIFF, m_interval))
                    smi_f9.append(get_feature_by_index(observation, Feature.PKT_INTERARRIVAL, m_interval))
                    smi_f10.append(get_feature_by_index(observation, Feature.PKT_JITTER, m_interval))
                    smi_f11.append(get_feature_by_index(observation, Feature.PKT_LOSS_RATIO, m_interval))
                    smi_f12.append(get_feature_by_index(observation, Feature.PKT_AVE_LOSS, m_interval))
                    smi_f13.append(get_feature_by_index(observation, Feature.VIDEO_PKT_PROB, m_interval))
                    smi_f14.append(get_feature_by_index(observation, Feature.AUDIO_PKT_PROB, m_interval))
                    smi_f15.append(get_feature_by_index(observation, Feature.PROB_PKT_PROB, m_interval))

                # add batch dimension for prediction
                observation = observation.reshape((1, len(observation))).astype(np.float32)
                prediction = algo.predict(observation)[0]
                predictions.append(prediction)

        bw_predictions = np.concatenate(bw_predictions)

        if self.__plot_log:
            f_rwds = np.append(f_rwds[1:], 0)
            #s_rwds = np.append(s_rwds[1:], 0)
            #l_rwds = np.append(l_rwds[1:], 0)

            # scaling
            scaler = MinMaxScaler(feature_range=(0,10))
            smi_f1 = scaler.fit_transform(np.array(smi_f1).reshape(-1,1))
            smi_f2 = scaler.fit_transform(np.array(smi_f2).reshape(-1,1))
            smi_f3 = scaler.fit_transform(np.array(smi_f3).reshape(-1,1))
            smi_f4 = scaler.fit_transform(np.array(smi_f4).reshape(-1,1))
            smi_f5 = scaler.fit_transform(np.array(smi_f5).reshape(-1,1))
            smi_f6 = scaler.fit_transform(np.array(smi_f6).reshape(-1,1))
            smi_f7 = scaler.fit_transform(np.array(smi_f7).reshape(-1,1))
            smi_f8 = scaler.fit_transform(np.array(smi_f8).reshape(-1,1))
            smi_f9 = scaler.fit_transform(np.array(smi_f9).reshape(-1,1))
            smi_f10 = scaler.fit_transform(np.array(smi_f10).reshape(-1,1))
            smi_f11 = scaler.fit_transform(np.array(smi_f11).reshape(-1,1))
            smi_f12 = scaler.fit_transform(np.array(smi_f12).reshape(-1,1))
            smi_f13 = scaler.fit_transform(np.array(smi_f13).reshape(-1,1))
            smi_f14 = scaler.fit_transform(np.array(smi_f14).reshape(-1,1))
            smi_f15 = scaler.fit_transform(np.array(smi_f15).reshape(-1,1))

        # plot the predictions
        x = range(len(predictions))
        predictions_scaled = [x / 1000000 for x in predictions]
        bw_predictions_scaled = [x / 1000000 for x in bw_predictions]


        if not self.__plot_log:
            plt.plot(x, predictions_scaled, label="estimate")
            plt.plot(x, bw_predictions_scaled, label="baseline")
            plt.legend()
            plt.ylabel("Estimate")
            plt.title('Estimate divided by 10e6')

            algo_name = self.__model_filename.split('/')[-1]
            log_file_name = filename.split('/')[-1]
            plt.xlabel(f'Evaluate {algo_name} on {log_file_name}')
            plt.show()

        else:
            plt.subplot(2, 4, 1)
            plt.plot(x, predictions_scaled, label="estimate")
            plt.plot(x, bw_predictions_scaled, label="baseline")
            plt.legend()
            plt.ylabel("Estimate")
            plt.xlabel("Step")
            plt.title('Estimate divided by 10e6')

            plt.subplot(2, 4, 5)
            plt.plot(x, f_rwds, label="t_reward")
            #plt.plot(x, s_rwds, label="s_reward")
            #plt.plot(x, l_rwds, label="l_reward")
            plt.legend()
            plt.ylabel('Reward')
            algo_name = self.__model_filename.split('/')[-1]
            log_file_name = filename.split('/')[-1]
            plt.xlabel(f'Evaluate {algo_name} on {log_file_name}')

            plt.subplot(2,4,2)
            plt.plot(x, smi_f1, label="f1-recv_rate")
            plt.plot(x, smi_f2, label="f2-recv_pkt_amount")
            plt.plot(x, smi_f3, label="f3-recv_bytes")
            plt.legend()

            plt.subplot(2,4,6)
            plt.plot(x, smi_f4, label="queuing_delay")
            plt.plot(x, smi_f5, label="delay")
            plt.plot(x, smi_f6, label="min_seen_delay")
            plt.legend()

            plt.subplot(2,4,3)
            plt.plot(x, smi_f7, label="delay_ratio")
            plt.plot(x, smi_f8, label="delay_min_diff")
            plt.plot(x, smi_f9, label="pkt_interarrival")
            plt.legend()

            plt.subplot(2,4,7)
            plt.plot(x, smi_f10, label="pkt_jitter")
            plt.plot(x, smi_f11, label="pkt_loss_ratio")
            plt.plot(x, smi_f12, label="pkt_ave_loss")
            plt.legend()

            plt.subplot(2,4,4)
            plt.plot(x, smi_f13, label="video_pkt_prob")
            plt.plot(x, smi_f14, label="audio_pkt_prob")
            plt.plot(x, smi_f15, label="prob_pkt_prob")
            plt.legend()

            r_preds = np.corrcoef(np.array(predictions_scaled).reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Model-Baseline corrcoef:\n {r_preds[0][1]}')

            r_pred_f1 = np.corrcoef(smi_f1.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Recv_rate corrcoef:\n {r_pred_f1[0][1]}')

            r_pred_f2 = np.corrcoef(smi_f2.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Recv_pkt_amount corrcoef:\n {r_pred_f2[0][1]}')

            r_pred_f3 = np.corrcoef(smi_f3.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Recv_bytes corrcoef:\n {r_pred_f3[0][1]}')

            r_pred_f4 = np.corrcoef(smi_f4.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Queuing_delay corrcoef:\n {r_pred_f4[0][1]}')

            r_pred_f5 = np.corrcoef(smi_f5.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Delay corrcoef:\n {r_pred_f5[0][1]}')

            r_pred_f6 = np.corrcoef(smi_f6.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Min_seen_delay corrcoef:\n {r_pred_f6[0][1]}')

            r_pred_f7 = np.corrcoef(smi_f7.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Delay_ratio corrcoef:\n {r_pred_f7[0][1]}')

            r_pred_f8 = np.corrcoef(smi_f8.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Delay_min_diff corrcoef:\n {r_pred_f8[0][1]}')

            r_pred_f9 = np.corrcoef(smi_f9.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Pkt_interarrival corrcoef:\n {r_pred_f9[0][1]}')

            r_pred_f10 = np.corrcoef(smi_f10.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Pkt_jitter corrcoef:\n {r_pred_f10[0][1]}')

            r_pred_f11 = np.corrcoef(smi_f11.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Pkt_loss_ratio corrcoef:\n {r_pred_f11[0][1]}')

            r_pred_f12 = np.corrcoef(smi_f12.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Pkt_ave_loss corrcoef:\n {r_pred_f12[0][1]}')

            r_pred_f13 = np.corrcoef(smi_f13.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Video_pkt_prob corrcoef:\n {r_pred_f13[0][1]}')

            r_pred_f14 = np.corrcoef(smi_f14.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Audio_pkt_prob corrcoef:\n {r_pred_f14[0][1]}')

            r_pred_f15 = np.corrcoef(smi_f15.reshape(-1), np.array(bw_predictions_scaled).reshape(-1))
            print(f'Baseline-Prob_pkt_prob corrcoef:\n {r_pred_f15[0][1]}')

            plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model", type=str)
    parser.add_argument("-p", "--parameters", type=str)
    parser.add_argument("-d", "--dir", type=str, default="emudata")
    parser.add_argument("-m", "--maxfiles", type=int, default=3)
    args = parser.parse_args()

    e = eval_model('..')
    e.run()


if __name__ == "__main__":
    main()
