from pathlib import Path

import my_flatbuffers as flatbuffers
import my_tflite as tflite
from my_flatbuffers import encode
from my_flatbuffers import number_types as N


def get_buf(path):
    with open(path, "rb") as f:
        buf = f.read()
    return buf


def get_model(path):
    model = tflite.Model.GetRootAsModel(get_buf(path), 0)
    return model


def get_metadata(path):
    with open(path, "rb") as f:
        buf = f.read()
        metadata = tflite.Metadata.GetRootAsMetadata(buf, 0)
    return metadata


def print_buf(buf):
    hex_str = buf.hex()
    for i in range(0, len(hex_str), 16):
        cur_hex = hex_str[i : i + 16]
        n = 2
        hex_list = []
        for i in range(0, len(cur_hex), n):
            hex_list.append(cur_hex[i : i + n])
        front_2 = "".join(hex_list[:2])
        front_4 = "".join(hex_list[2:4])
        backward_2 = "".join(hex_list[4:6])
        backward_4 = "".join(hex_list[6:])
        print(f"{front_2} {front_4} {backward_2} {backward_4}")


if __name__ == "__main__":
    fc1_path = Path("../HMSL-TPU-Desktop/3_tflite_analyze/models/temp/temp.tflite")
    fc1_tpu_path = Path("../HMSL-TPU-Desktop/3_tflite_analyze/models/temp/temp_edgetpu.tflite")

    fc1_model = get_model(fc1_path)
    fc1_tpu_model = get_model(fc1_tpu_path)

    print(fc1_model.Version())
    
