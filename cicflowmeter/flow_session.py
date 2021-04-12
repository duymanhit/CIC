import os
import csv
import numpy
import warnings
import ipaddress
import numpy as np
from .flow import Flow
from pickle import load
from collections import defaultdict
from scapy.sessions import DefaultSession
from pathlib import Path, PureWindowsPath
from tensorflow.keras.models import load_model
from .features.context.packet_direction import PacketDirection
from .features.context.packet_flow_key import get_packet_flow_key

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EXPIRED_UPDATE = 40
MACHINE_LEARNING_API = "http://localhost:8000/predict"


class FlowSession(DefaultSession):
    """Creates a list of network flows."""

    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.csv_line = 0

        if self.output_mode == "flow":
            output = open(self.output_file, "w")
            self.csv_writer = csv.writer(output)

        self.packets_count = 0

        self.clumped_flows_per_label = defaultdict(list)

        super(FlowSession, self).__init__(*args, **kwargs)

    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        self.garbage_collect(None)
        return super(FlowSession, self).toPacketList()

    def on_packet_received(self, packet):
        count = 0
        direction = PacketDirection.FORWARD

        if self.output_mode != "flow":
            if "IP" not in packet:
                return

        self.packets_count += 1

        # Creates a key variable to check
        packet_flow_key = get_packet_flow_key(packet, direction)
        flow = self.flows.get((packet_flow_key, count))

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))

            if flow is None:
                # If no flow exists create a new flow
                direction = PacketDirection.FORWARD
                flow = Flow(packet, direction)
                packet_flow_key = get_packet_flow_key(packet, direction)
                self.flows[(packet_flow_key, count)] = flow

            elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
                # If the packet exists in the flow but the packet is sent
                # after too much of a delay than it is a part of a new flow.
                expired = EXPIRED_UPDATE
                while (packet.time - flow.latest_timestamp) > expired:
                    count += 1
                    expired += EXPIRED_UPDATE
                    flow = self.flows.get((packet_flow_key, count))

                    if flow is None:
                        flow = Flow(packet, direction)
                        self.flows[(packet_flow_key, count)] = flow
                        break

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            expired = EXPIRED_UPDATE
            while (packet.time - flow.latest_timestamp) > expired:

                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))

                if flow is None:
                    flow = Flow(packet, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break

        flow.add_packet(packet, direction)

        if self.packets_count % 100 == 0 or (
            flow.duration > 120 and self.output_mode == "flow"
        ):
            # print("Packet count: {}".format(self.packets_count))
            self.garbage_collect(packet.time)

    def get_flows(self) -> list:
        return self.flows.values()

    def garbage_collect(self, latest_time) -> None:
        # TODO: Garbage Collection / Feature Extraction should have a separate thread
        # print("Garbage Collection Began. Flows = {}".format(len(self.flows)))
        keys = list(self.flows.keys())
        for k in keys:
            flow = self.flows.get(k)

            if (
                latest_time is None
                or latest_time - flow.latest_timestamp > EXPIRED_UPDATE
                or flow.duration > 90
            ):
                data = flow.get_data()
                data = predict(data)

                # POST Request to Model API
                # if self.url_model:
                #     payload = {
                #         "columns": list(data.keys()),
                #         "data": [list(data.values())],
                #     }
                #     post = requests.post(
                #         self.url_model,
                #         json=payload,
                #         headers={
                #             "Content-Type": "application/json; format=pandas-split"
                #         },
                #     )
                #     resp = post.json()
                #     result = resp["result"].pop()
                #     if result == 0:
                #         # benign_threshold = 0.9
                #         # if resp["probability"][0][result] < benign_threshold:
                #         #     result_print = "Malicious"
                #         # else:
                #         #     result_print = "Benign"
                #         result_print = "Benign"
                #     else:
                #         result_print = "Malicious"



                if self.csv_line == 0:
                    self.csv_writer.writerow(data.keys())

                self.csv_writer.writerow(data.values())
                self.csv_line += 1
                #print("CIC: " + str(data))



                del self.flows[k]
        # print("Garbage Collection Finished. Flows = {}".format(len(self.flows)))


def predict(data):
    #data = request.get_json()
    features = ['flow_duration', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_std', 'flow_iat_mean',
                'flow_iat_max', 'fwd_iat_mean', 'fwd_iat_max', 'fwd_header_len', 'fwd_pkts_s', 'pkt_len_min',
                'pkt_len_max', 'pkt_len_std', 'ack_flag_cnt', 'pkt_size_avg', 'fwd_header_len', 'subflow_fwd_byts',
                'init_fwd_win_byts', 'fwd_seg_size_min', ]
    labels = ['DNS', 'LDAP', 'MSSQL', 'NTP', 'NetBIOS', 'Portmap', 'SNMP', 'SSDP', 'Syn', 'TFTP', 'UDP', 'UDPLag',
              'WebDDoS']

    xTest = []
    for feature in features:
        if feature in data:
            try:
                x = float(data[feature])
                xTest.append(data[feature])
            except:
                warnings.warn("Invalid type: Feature "+feature+ " = "+data[feature]+" has type "+type(data[feature]))

    binary_model_path = PureWindowsPath('model/output/ML_binary_classifier_model')
    binary_model_path = Path(binary_model_path)

    category_model_path = PureWindowsPath('model/output/ML_category_classifier_model')
    category_model_path = Path(category_model_path)

    scaler_path = PureWindowsPath('model/scaler.pkl')
    scaler_path = Path(scaler_path)
    scaler = load(open(scaler_path, 'rb'))
    xTest = scaler.transform(numpy.array(xTest).reshape(1, -1))


    model = load_model(binary_model_path)

    prob = model(xTest).numpy()
    prediction = np.argmax(model(xTest), axis=1)
    if prediction == 0:
        data['MALICIOUS'] = None
        for i in range(len(labels)):
            data[labels[i]] = None
        return data
    elif prediction == 1:
        print("Phát hiện tấn công: "+ str(round(prob[0][1]*100,2)) + "%")

        data['MALICIOUS'] = prob[0][1]
        model = load_model(category_model_path)
        prob = model(xTest).numpy()
        prediction = np.argmax(model(xTest), axis=1)

        if not len(prediction) == 1:
            warnings.warn("NOT SINGLE PREDICTION")
            return data


        for i in range(len(labels)):
            data[labels[i]] = prob[0][i]
        print('\x1b[6;30;42m' + labels[prediction[0]] + '\x1b[0m')

        probs = prob[0]
        n = len(labels)
        idx = np.argsort(probs)[::-1][:n]

        probs = np.array(probs)[idx]
        labels = np.array(labels)[idx]

        for i in range(len(labels)):
            print(labels[i]+': '+str(round(probs[i]*100,2))+'%',end ="  ")
        print()
        print(data)
        print("**********************")



    return data

def generate_session_class(output_mode, output_file, url_model):
    return type(
        "NewFlowSession",
        (FlowSession,),
        {
            "output_mode": output_mode,
            "output_file": output_file,
            "url_model": url_model,
        },
    )
