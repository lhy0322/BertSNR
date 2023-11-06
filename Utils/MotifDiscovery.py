import numpy as np
from collections import Counter

def MotifDiscovery(DNASequence, PredictLabel, WindowLength):
    Result = ''
    PredictLabel = np.array(PredictLabel)  # shape = (100,)      shape[0] = 100
    ContainMotif = 0  # 0 表示没有Motif     1 表示有Motif

    WindowLeftPosition = 0
    WindowRightPosition = WindowLeftPosition + WindowLength

    FinalLeftFlag = 0

    PriorStatus = 0  # 表示上次窗口中有或者没有Motif
    CurrentStatus = 0  # 表示当前窗口中有或者没有Motif

    FinalLeftPosition = 0
    FinalRightPosition = 0
    FinalCenter = 0

    while WindowRightPosition <= 100:
        PositiveNumber = Counter(PredictLabel[WindowLeftPosition: WindowRightPosition])[1]  # 统计Mer中含有的1的个数
        if PositiveNumber >= WindowLength * 0.8:
            ContainMotif = 1
            CurrentStatus = 1
            FinalRightPosition = WindowRightPosition
            if FinalLeftFlag == 0:
                FinalLeftPosition = WindowLeftPosition
                FinalLeftFlag = 1
        else:
            CurrentStatus = 0

        WindowLeftPosition = WindowLeftPosition + 1
        WindowRightPosition = WindowRightPosition + 1

        if PriorStatus == 1 and CurrentStatus == 0:
            break

        PriorStatus = CurrentStatus

    # 没有Motif的情况
    if ContainMotif == 0:
        Result = 'N' * WindowLength

    # 有Motif的情况
    if ContainMotif == 1:
        FinalCenter = int((FinalLeftPosition + FinalRightPosition) / 2)  # int 表示向下取整
        Result = DNASequence[FinalCenter - (WindowLength//2): FinalCenter + (WindowLength//2) + 1]
    return Result