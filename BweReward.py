import numpy as np
from typing import Any, Dict, List

from BweUtils import get_decay_weights

"""
Features list:

1. Receiving rate: rate at which the client receives data from the sender during a MI, unit: bps.
2. Number of received packets: total number of packets received in a MI, unit: packet.
3. Received bytes: total number of bytes received in a MI, unit: Bytes.
4. Queuing delay: average delay of packets received in a MI minus the minimum packet delay observed so far, unit: ms.
5. Delay: average delay of packets received in a MI minus a fixed base delay of 200ms, unit: ms.
6. Minimum seen delay: minimum packet delay observed so far, unit: ms.
7. Delay ratio: average delay of packets received in a MI divided by the minimum delay of packets received in the same MI, unit: ms/ms.
8. Delay average minimum difference: average delay of packets received in a MI minus the minimum delay of packets received in the same MI, unit: ms.
9. Packet interarrival time: mean interarrival time of packets received in a MI, unit: ms.
10. Packet jitter: standard deviation of interarrival time of packets received in a MI, unit: ms.
11. Packet loss ratio: probability of packet loss in a MI, unit: packet/packet.
12. Average number of lost packets: average number of lost packets given a loss occurs, unit: packet.
13. Video packets probability: proportion of video packets in the packets received in a MI, unit: packet/packet.
14. Audio packets probability: proportion of audio packets in the packets received in a MI, unit: packet/packet.
15. Probing packets probability: proportion of probing packets in the packets received in a MI, unit: packet/packet.

The indices (zero-indexed) of features over the 5 short term MIs are
{(feature number - 1) * 10, ., ., ., (feature number - 1) * 10 + 4}.

The indices (zero-indexed) of features over the 5 long term MIs are
{(feature number - 1) * 10 + 5, ., ., ., (feature number - 1) * 10 + 9}.

The rule: i + 150 * (j - 1), where i is the feature index (0 -- 149) and j is the iteration index (1 - ..).

First five are for short term, the rest are for long term.

1. Receiving rate: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
2. Number of received packets: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
3. Received bytes: 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
4. Queuing delay: 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
5. Delay: 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
6. Minimum seen delay: 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
7. Delay ratio: 60, 61, 62, 63, 64, 65, 66, 67, 68, 69
8. Delay average minimum difference: 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
9. Packet interarrival time: 80, 81, 82, 83, 84, 85, 86, 87, 88, 89
10. Packet jitter: 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
11. Packet loss ratio: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109
12. Average number of lost packets: 110, 111, 112, 113, 114, 115, 116, 117, 118, 119
13. Video packets probability: 120, 121, 122, 123, 124, 125, 126, 127, 128, 129
14. Audio packets probability: 130, 131, 132, 133, 134, 135, 136, 137, 138, 139
15. Probing packets probability: 140, 141, 142, 143, 144, 145, 146, 147, 148, 149

# timesteps

## short term, indicies: 0, 1, 2, 3, 4

-60 ms back, -120 ms back, -180 ms back, -240 ms back, -300 ms back

## long term, indicies: 5, 6, 7, 8, 9

-600 ms back, -1200 ms back, -1800 ms back, -2400 ms back, -3000 ms back

## distance between two consecutive observations:

60 ms

## the last timestamp is the last packet's timestamp, so a small shift could be observed 

"""

from enum import IntEnum


class Feature(IntEnum):
    RECV_RATE = 1
    RECV_PKT_AMOUNT = 2
    RECV_BYTES = 3
    QUEUING_DELAY = 4
    DELAY = 5
    MIN_SEEN_DELAY = 6
    DELAY_RATIO = 7
    DELAY_MIN_DIFF = 8
    PKT_INTERARRIVAL = 9
    PKT_JITTER = 10
    PKT_LOSS_RATIO = 11
    PKT_AVE_LOSS = 12
    VIDEO_PKT_PROB = 13
    AUDIO_PKT_PROB = 14
    PROB_PKT_PROB = 15


class MI(IntEnum):
    SHORT_60 = 1
    SHORT_120 = 2
    SHORT_180 = 3
    SHORT_240 = 4
    SHORT_300 = 5
    LONG_600 = 6
    LONG_1200 = 7
    LONG_1800 = 8
    LONG_2400 = 9
    LONG_3000 = 10


class MIType(IntEnum):
    SHORT = 1
    LONG = 2
    ALL = 3


def get_feature(
    observation: List[float],
    feature_name: str,
    mi: MI = MI.LONG_600,
) -> float:
    match (feature_name.upper()):
        case "RECV_RATE":
            return observation[(Feature.RECV_RATE - 1) * 10 + mi - 1]
        case "RECV_PKT_AMOUNT":
            return observation[(Feature.RECV_PKT_AMOUNT - 1) * 10 + mi - 1]
        case "RECV_BYTES":
            return observation[(Feature.RECV_BYTES - 1) * 10 + mi - 1]
        case "QUEUING_DELAY":
            return observation[(Feature.QUEUING_DELAY - 1) * 10 + mi - 1]
        case "DELAY":
            return observation[(Feature.DELAY - 1) * 10 + mi - 1]
        case "MIN_SEEN_DELAY":
            return observation[(Feature.MIN_SEEN_DELAY - 1) * 10 + mi - 1]
        case "DELAY_RATIO":
            return observation[(Feature.DELAY_RATIO - 1) * 10 + mi - 1]
        case "DELAY_MIN_DIFF":
            return observation[(Feature.DELAY_MIN_DIFF - 1) * 10 + mi - 1]
        case "PKT_INTERARRIVAL":
            return observation[(Feature.PKT_INTERARRIVAL - 1) * 10 + mi - 1]
        case "PKT_JITTER":
            return observation[(Feature.PKT_JITTER - 1) * 10 + mi - 1]
        case "PKT_LOSS_RATIO":
            return observation[(Feature.PKT_LOSS_RATIO - 1) * 10 + mi - 1]
        case "PKT_AVE_LOSS":
            return observation[(Feature.PKT_AVE_LOSS - 1) * 10 + mi - 1]
        case "VIDEO_PKT_PROB":
            return observation[(Feature.VIDEO_PKT_PROB - 1) * 10 + mi - 1]
        case "AUDIO_PKT_PROB":
            return observation[(Feature.AUDIO_PKT_PROB - 1) * 10 + mi - 1]
        case "PROB_PKT_PROB":
            return observation[(Feature.PROB_PKT_PROB - 1) * 10 + mi - 1]
        case _:
            raise ValueError(f"Unknown feature name: {feature_name}")


def get_feature_for_mi(
    observation: List[float],
    feature_name: str,
    mi_type: MIType = MIType.ALL,
) -> List[float]:
    match (mi_type):
        case MIType.SHORT:
            return [get_feature(observation, feature_name, mi) for mi in MI if mi <= MI.SHORT_300]
        case MIType.LONG:
            return [get_feature(observation, feature_name, mi) for mi in MI if mi >= MI.LONG_600]
        case MIType.ALL:
            return [get_feature(observation, feature_name, mi) for mi in MI]
        case _:
            raise ValueError(f"Unknown MI type: {mi_type}")


#######################################################################################################################
class RewardFunction:
    def __init__(self, reward_func_name: str = "QOE_V1"):
        self.inner_params = {}
        match (reward_func_name.upper()):
            case "BWE":
                self.reward_func = reward_bwe
            case "R3NET":
                self.reward_func = reward_r3net
            case "ONRL":
                self.reward_func = reward_onrl
            case "QOE_V1":
                self.reward_func = reward_qoe_v1
                self.inner_params = {
                    "MAX_RATE": dict(zip(MI, [0.0] * len(MI))),
                    "MAX_DELAY": dict(zip(MI, [0.0] * len(MI))),
                }
            case _:
                raise ValueError(f"Unknown reward function name: {reward_func_name}")

    def __call__(self, observation: List[float]) -> float:
        return self.reward_func(observation, self.inner_params)


# reward function {0.6 * ln(4R + 1) -D - 10L} (inspired from R3Net)
# R - receive rate at this time step, mb/s
# D - queuing delay, ms (R3Net uses average Round Trip Time)
# L - packet loss rate
# receiving more packets is rewarded, delay and packet loss is penalized.
def reward_r3net(feature_vec: List[float], rf_params: Dict[str, Any]=None) -> float:
    # use the 5 recent short MIs. (TODO: refinement)
    receive_rate = np.sum(feature_vec[(Feature.RECV_RATE - 1) * 10 : (Feature.RECV_RATE - 1) * 10 + 5]) / 5
    queuing_delay = np.sum(feature_vec[(Feature.QUEUING_DELAY - 1) * 10 : (Feature.QUEUING_DELAY - 1) * 10 + 5]) / 5
    pkt_loss_rate = np.sum(feature_vec[(Feature.PKT_LOSS_RATIO - 1) * 10 : (Feature.PKT_LOSS_RATIO - 1) * 10 + 5]) / 5

    return 0.6 * np.log(4 * receive_rate + 1) - queuing_delay / 1000 - 10 * pkt_loss_rate


# reward function (inspired from OnRL)
# definition from OnRL follows
# r_t = alpha * sum_{n=1}^{N} q_n
#       - beta * sum_{n=1}^{N} l_n
#       - eta * sum_{n=1}^{N} d_n
#       - phi * sum_{n=1}^{N-1} |q_n - q_{n-1}|
# N - number of RTP packets in a state
# q_n - receive rate at this time step, mb/s
# l_n - packet loss rate
# d_n - delay, ms
# |q_n - q_{n-1}| - receive rates difference between two steps. ()
# alpha, beta, eta and phi are hyperparameters impacting training effectiveness.
#
# Here we uses the statistic of recent measure intervals (as the feature vectors)
#
def reward_onrl(feature_vec: List[float], rf_params: Dict[str, Any]=None) -> float:
    alpha = 50
    beta = 50
    eta = 10
    phi = 20

    receive_rate = np.sum(feature_vec[(Feature.RECV_RATE - 1) * 10 : (Feature.RECV_RATE - 1) * 10 + 5])/5
    queuing_delay = np.sum(feature_vec[(Feature.QUEUING_DELAY - 1) * 10 : (Feature.QUEUING_DELAY - 1) * 10 + 5])/5
    pkt_loss_rate = np.sum(feature_vec[(Feature.PKT_LOSS_RATIO - 1) * 10 : (Feature.PKT_LOSS_RATIO - 1) * 10 + 5])/5

    # incomplete - the last term is not in place. (TODO)
    return alpha * receive_rate - beta * pkt_loss_rate - eta * queuing_delay - 0


# Reward function from my paper with QoE ideas
# 1. Rate QoE has a logarithmic nature because at some high point on the user will not be able to distinguish and at some low point the user will not be satisfied.
# it should be always scaled with the maximum possible rate (could be a running or fixed value), then it is pretty generic.
# 2. Delay QoE is computed by dividing the current performance to the best observed and by exploring the current average value. It is simply d_max - d / (d_max - d_min)
# 3. Loss QoE is simply a linear function of the loss rate, I don't know any better way to compute it.
# 4. Jitter has a square-root nature, I took it from this paper: https://ieeexplore.ieee.org/document/9148101
# Other features are not used for the reward function, but they are used for the state representation.
def reward_qoe_v1(observation: List[float], rf_params: Dict[str, Any]) -> float:
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
    return final_qoe


def reward_bwe(observation: List[float], rf_params: Dict[str, Any]=None) -> float:

    mi_cur = MI.LONG_600
    mi_nxt = MI.LONG_1200
    mi_list = [mi_cur, mi_nxt]
    rewards = []
    final_reward = 0.0

    for mi in mi_list:
        # to award
        video_pkt_prob = np.sum(observation[(Feature.VIDEO_PKT_PROB - 1) * 10 + mi: (Feature.VIDEO_PKT_PROB - 1) * 10 + 5 + mi]) / 5
        receive_rate = np.sum(observation[(Feature.RECV_RATE - 1) * 10 + mi: (Feature.RECV_RATE - 1) * 10 + 5 + mi]) / 5

        award = (0.5*video_pkt_prob + 0.5*receive_rate)

        # to punish
        audio_pkt_prob = np.sum(observation[(Feature.AUDIO_PKT_PROB - 1) * 10 + mi: (Feature.AUDIO_PKT_PROB - 1) * 10 + 5 + mi]) / 5
        pkt_interarrival = np.sum(observation[(Feature.PKT_INTERARRIVAL - 1) * 10 + mi : (Feature.PKT_INTERARRIVAL - 1) * 10 + 5 + mi]) / 5
        pkt_jitter = np.sum(observation[(Feature.PKT_JITTER - 1) * 10 + mi: (Feature.PKT_JITTER - 1) * 10 + 5 + mi]) / 5
        pkt_loss_rate = np.sum(observation[(Feature.PKT_LOSS_RATIO - 1) * 10 + mi: (Feature.PKT_LOSS_RATIO - 1) * 10 + 5 + mi]) / 5
        queuing_delay = np.sum(observation[(Feature.QUEUING_DELAY - 1) * 10 + mi: (Feature.QUEUING_DELAY - 1) * 10 + 5 + mi]) / 5

        fine = (0.35*audio_pkt_prob + 0.35*pkt_interarrival + 0.1*pkt_jitter + 0.1*pkt_loss_rate + 0.1*queuing_delay)

        rewards.append((0.7*award - 0.5*fine)*5)

    diff = (rewards[1] - 0.5*rewards[0])
    diff_per = np.abs(diff)/np.abs(rewards[0])
    alpha = 0.6
    beta = 0.4
    theta = 0.3
    if diff_per <= 0.1:
        final_reward = rewards[1]
    elif 0.1 < diff_per <= 0.3:
        final_reward = rewards[0] * (1.1 + alpha * (diff_per - 0.1))
    elif 0.3 < diff_per <= 0.6:
        final_reward = rewards[0] * (1.3 + beta * (diff_per - 0.3))
    else:
        final_reward = rewards[0] * (1.6 + alpha * (diff_per - 0.6))

    return final_reward