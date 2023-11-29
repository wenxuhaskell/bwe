import numpy as np

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

"""

from enum import IntEnum


class Feature(IntEnum):
    RECV_RATE = 1
    RECV_PKT_AMOUNT = 2
    RECV_BYTEs = 3
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


# reward function {0.6 * ln(4R + 1) -D - 10L} (inspired from R3Net)
# R - receive rate at this time step, mb/s
# D - queuing delay, ms (R3Net uses average Round Trip Time)
# L - packet loss rate
# receiving more packets is rewarded, delay and packet loss is penalized.

def reward_r3net(feature_vec):
    # use the 5 recent short MIs. (TODO: refinement)
    receive_rate = np.sum(feature_vec[(Feature.RECV_RATE - 1) * 10: (Feature.RECV_RATE - 1) * 10 + 5]) / 5
    queuing_delay = np.sum(feature_vec[(Feature.QUEUING_DELAY - 1) * 10: (Feature.QUEUING_DELAY - 1) * 10 + 5]) / 5
    pkt_loss_rate = np.sum(feature_vec[(Feature.PKT_LOSS_RATIO - 1) * 10: (Feature.PKT_LOSS_RATIO - 1) * 10 + 5]) / 5

    return (0.6 * np.log(4 * receive_rate + 1) - queuing_delay / 1000 - 10 * pkt_loss_rate)


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
def reward_onrl(feature_vec):
    alpha = 50
    beta = 50
    eta = 10
    phi = 20

    receive_rate = np.sum(feature_vec[(Feature.RECV_RATE - 1) * 10: (Feature.RECV_RATE - 1) * 10 + 5])
    queuing_delay = np.sum(feature_vec[(Feature.QUEUING_DELAY - 1) * 10: (Feature.QUEUING_DELAY - 1) * 10 + 5])
    pkt_loss_rate = np.sum(feature_vec[(Feature.PKT_LOSS_RATIO - 1) * 10:(Feature.PKT_LOSS_RATIO - 1) * 10 + 5])

    # incomplete - the last term is not in place. (TODO)
    return (alpha * receive_rate - beta * pkt_loss_rate - eta * queuing_delay - 0)
