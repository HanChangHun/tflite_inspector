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
        cur_hex = hex_str[i: i + 16]
        n = 2
        hex_list = []
        for i in range(0, len(cur_hex), n):
            hex_list.append(cur_hex[i: i + n])
        front_2 = "".join(hex_list[:2])
        front_4 = "".join(hex_list[2:4])
        backward_2 = "".join(hex_list[4:6])
        backward_4 = "".join(hex_list[6:])
        print(f"{front_2} {front_4} {backward_2} {backward_4}")


if __name__ == "__main__":
    fc1_path = Path("models/temp.tflite")
    fc1_model = get_model(fc1_path)

    # fc1_tpu_path = Path("models/temp_edgetpu.tflite")
    # fc1_tpu_model = get_model(fc1_tpu_path)

    # fc1_model_Version = fc1_model.Version()
    # fc1_model_OperatorCodes = fc1_model.OperatorCodes(0)
    # fc1_model_OperatorCodesLength = fc1_model.OperatorCodesLength()
    # fc1_model_Description = fc1_model.Description()
    # fc1_model_Subgraphs = fc1_model.Subgraphs(0)
    # fc1_model_Subgraphs = fc1_model.Subgraphs(1)
    # fc1_model_Subgraphs = fc1_model.Subgraphs(2)
    # fc1_model_SubgraphsLength = fc1_model.SubgraphsLength()
    # fc1_model_Buffers = fc1_model.Buffers(0)
    # fc1_model_Buffers = fc1_model.Buffers(1)
    # fc1_model_Buffers = fc1_model.Buffers(2)
    # fc1_model_BuffersLength = fc1_model.BuffersLength()
    # fc1_model_Metadata = fc1_model.Metadata(0)
    # fc1_model_MetadataLength = fc1_model.MetadataLength()
    # fc1_model_SignatureDefs = fc1_model.SignatureDefs(0)
    # fc1_model_SignatureDefsLength = fc1_model.SignatureDefsLength()

    for i in range(fc1_model.SubgraphsLength()):
        print("=" * 20 + f"Subgraphs {i}" + "=" * 20)
        fc1_model.Subgraphs(i).Tensors(0)
        fc1_model.Subgraphs(i).Inputs(0)
        fc1_model.Subgraphs(i).Outputs(0)
    
    for i in range(fc1_model.BuffersLength()):
        print("=" * 20 + f"Buffers {i}" + "=" * 20)
        fc1_model.Buffers(i).DataAsNumpy()
