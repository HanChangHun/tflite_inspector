# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import my_flatbuffers as flatbuffers

class RankOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRankOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RankOptions()
        x.Init(buf, n + offset)
        return x

    # RankOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def RankOptionsStart(builder): builder.StartObject(0)
def RankOptionsEnd(builder): return builder.EndObject()
