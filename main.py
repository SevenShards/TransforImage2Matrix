import threading

from PIL import Image
import numpy as np
import os
import time
from functools import wraps
import threading
import queue

image_dir = './images/'
result_dir = './result/'
array_file = './array.bin'
rbgBuffer = './buffer/'
matrixBuffer = './matrixBuffer/'

img_length = 0
img_width = 0

basebuffer = {}
signalbuffer = {}
channelbuffer = {}


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.func_name, str(t1 - t0))
              )
        return result

    return function_timer


'''
    将图片按照分辨率大小
    分割成R,G,B三个通道的像素矩阵
    并保存
'''


def Image_to_array_file():
    filenames = os.listdir(image_dir)
    print(filenames)
    for filename in filenames:
        img = Image.open(image_dir + filename)
        global img_length
        img_length = img.size[0]
        global img_width
        img_width = img.size[1]

        # img.show()
        r, g, b = img.split()
        print('type(r): ', type(r), r)

        # print(r)
        r_matrix = np.array(r).reshape(img_length, img_width)
        print('type(r_matrix):', type(r_matrix), np.shape(r_matrix))
        g_matrix = np.array(g).reshape(img_length, img_width)
        b_matrix = np.array(b).reshape(img_length, img_width)
        # print(r_arr)

        np.savetxt("./buffer/r.txt", r_matrix, fmt='%d', delimiter=',')
        np.savetxt("./buffer/g.txt", g_matrix, fmt='%d', delimiter=',')
        np.savetxt("./buffer/b.txt", b_matrix, fmt='%d', delimiter=',')
        return r_matrix, g_matrix, b_matrix


'''
    读取近似处理后的R,G,B三通道的像素
    并使用pilow库合并成新的图片
'''


def ImageToMatrix():
    # r = Image.fromarray(np.loadtxt("./buffer/r_restore.txt", delimiter=',').astype(np.uint8).reshape(720, 1280))
    # g = Image.fromarray(np.loadtxt("./buffer/g_restore.txt", delimiter=',').astype(np.uint8).reshape(720, 1280))
    # b = Image.fromarray(np.loadtxt("./buffer/b_restore.txt", delimiter=',').astype(np.uint8).reshape(720, 1280))

    r = Image.fromarray(channelbuffer.get("r_restore").astype(np.uint8).reshape(720, 1280))
    g = Image.fromarray(channelbuffer.get("g_restore").astype(np.uint8).reshape(720, 1280))
    b = Image.fromarray(channelbuffer.get("b_restore").astype(np.uint8).reshape(720, 1280))

    # print('r = np.loadtxt: ', type(r), r)
    # print('g = np.loadtxt: ', type(g), g)
    # print('b = np.loadtxt: ', type(b), b)

    # r_temp = Image.fromarray(np.loadtxt("./buffer/r.txt", delimiter=',').astype(np.uint8).reshape(720, 1280))
    # print('r_temp = np.loadtxt: ', type(r_temp), r_temp)

    temp = [r, g, b]
    img = Image.merge('RGB', temp)
    img.show()
    img.save('./images/test1.jpeg', 'JPEG')
    print(r)


'''
    迭代近似函数
'''


def DistilingMatrix(channelName, flagNumber, delta, pixelMatrix):
    min = pixelMatrix[0][0]
    max = pixelMatrix[0][0]
    # print('*********************')
    # print(min,max)
    mid = 0
    # calculate max and min value
    for i in np.nditer(pixelMatrix, order='C'):
        if i >= max:
            max = i
        if i <= min:
            min = i
    # print("max min: ", max, min)
    delta = int(max) - int(min)
    # print("delta: ", delta)
    # calculate mid value
    if delta % 2 != 0:
        mid = (int(max) + int(min) + 1) / 2
    else:
        mid = (int(max) + int(min)) / 2
    # print("mid, type(mid): ", mid, type(mid))
    # creat base matrix b1
    b = (np.ones((16, 16)) * mid).astype('int32')
    # save the baseMatrix in turns
    global basebuffer
    basebuffer.update({channelName+"_b" + str(flagNumber): b})
    # b.tofile(matrixBuffer + "b" + str(flagNumber) + ".bin")
    # print("*" * 100)
    '''
    print base matrix
    '''
    # print("base matrix \n", b)

    # calculate delta matrix d1
    d = (pixelMatrix - mid).astype('int32')
    '''
        print delta matrix
    '''
    # print("delta matrix: \n", d)

    # create absolute matrix a1 and signal matrix s1
    a = np.empty([16, 16], dtype="int")
    s = np.empty([16, 16], dtype="int")
    # print("*" * 100)
    # print(type(d1))
    for x in range(d.shape[0]):
        for y in range(d.shape[1]):
            if (d[x][y] < 0):
                a[x][y] = d[x][y] * (-1)
                s[x][y] = -1
            else:
                a[x][y] = d[x][y]
                s[x][y] = 1
    # save the signal Matrix in turns
    global signalbuffer
    signalbuffer.update({channelName+"_s" + str(flagNumber): s})
    # s.tofile(matrixBuffer + "s" + str(flagNumber) + ".bin")
    '''
        print signal matrix
    '''
    # print("signal matrix: \n", s)
    return delta, a


'''
    按照迭代的过程，将每一轮重新相乘得到的所有矩阵与第一个Base矩阵相加
'''


def reconstructMatrix(channelName, loopNUmber):
    # b_reconstructMatrix = np.fromfile("../TransforImage2Matrix/matrixBuffer/b1.bin", dtype=np.int32).reshape(16, 16)
    b_reconstructMatrix = basebuffer.get(channelName+"_b1")
    for i in range(loopNUmber, 0, -1):
        if i == 1:
            break
        # print(i)
        b_temp = readBaseBin(channelName, i)

        b_reconstructMatrix = b_reconstructMatrix + b_temp
        # print("b_temp: \n", b_temp)

    '''
            print base reconstruct matrix
    '''
    # print("b_reconstructMatrix: \n", b_reconstructMatrix)
    return b_reconstructMatrix


'''
    按照迭代的次数
    舍去最后一个delta矩阵，并与所有之前迭代的signal矩阵相乘，得到该轮处理后的base矩阵
'''


def readBaseBin(channelName, number):
    # print(number, '\n')
    b = basebuffer.get(channelName+"_b" + str(number))
    # b = np.fromfile("./matrixBuffer/b" + str(number) + ".bin", dtype=np.int32).reshape(16, 16)
    # print("BaseNumber: ", number, "BaseMatrix: \n" , b)
    for i in range(number, 0, -1):
        if i == 1:
            break
        s = readSignalBin(channelName, i - 1)
        b = b * s

    # print("b: \n", b)
    return b


'''
    获取基于base矩阵的前一轮的signal矩阵
'''


def readSignalBin(channelName, number):
    s = signalbuffer.get(channelName+"_s" + str(number))
    # s = np.fromfile("./matrixBuffer/s" + str(number) + ".bin", dtype=np.int).reshape(16, 16)
    # print("SignalNumber: ", number, "SignalMatrix: \n", s)
    return s


'''
    从水平方向堆叠16x16矩阵，直到16x720为止
'''


def restorePixelMatrixInHorizontal(restore_pixel_matrix_horizontal_temp, b_reconstructMatrix):
    restore_pixel_matrix_horizontal = np.hstack((restore_pixel_matrix_horizontal_temp, b_reconstructMatrix))
    return restore_pixel_matrix_horizontal


'''
    将16x720矩阵在垂直方向堆叠，直到1280x720为止
'''


def restorePixelMatrixInVertical(restore_pixel_matrix_temp, restore_pixel_matrix_horizontal_temp):
    restore_pixel_matrix = np.vstack((restore_pixel_matrix_temp, restore_pixel_matrix_horizontal_temp))
    return restore_pixel_matrix


'''
    对读取到的各个通道的像素矩阵开始分割成16x16的矩阵
    并执行迭代近似过程，最后将近似后的矩阵按通道名进行保存
'''


def DistilingProgram(pixelChanneMatrix, channelName):
    # ImageToMatrix()

    '''
    separate keyframe pixel matrix to 16x16 in r,g,b channel respective
    '''

    restore_pixel_matrix = None
    restore_pixel_matrix_horizontal_temp = None

    # 1280x720 按行分割成80个16x720的矩阵
    pixelChannelMatrix_vertical = np.vsplit(pixelChanneMatrix, img_length / 16)
    # print(np.shape(pixelChannelMatrix_vertical), pixelChannelMatrix_vertical)
    # (80, 16, 720)
    count_length = 0
    vertical_flag = True
    # 共80组  45x[16,16]矩阵,从第一组开始循环执行
    for pixelChannelMatrix_vertical_temp in pixelChannelMatrix_vertical:

        # 循环按列将每一行分割成45个16x16的矩阵
        pixelChanneMatrix_vertical_temp_horizontal = np.hsplit(pixelChannelMatrix_vertical_temp, img_width / 16)
        # (45,16,16)
        # print(np.shape(pixelChanneMatrix_vertical_temp_horizontal))
        count_width = 0
        horizontal_flag = True
        # 开始对第一个16x16执行Distiling,共计45个
        for pixelChanneMatrix_vertical_temp_horizontal_original in pixelChanneMatrix_vertical_temp_horizontal:

            # if count_1 == 1:
            '''
                print original matrix
            '''
            # print("original matrix: \n", pixelChanneMatrix_vertical_temp_horizontal_original)
            # print("数据类型", type(pixelChanneMatrix_vertical_temp_horizontal_original))  # 打印数组数据类型
            # print("数组元素数据类型：", pixelChanneMatrix_vertical_temp_horizontal_original.dtype)  # 打印数组元素数据类型
            min = pixelChanneMatrix_vertical_temp_horizontal_original[0][0]
            max = min
            for i in np.nditer(pixelChanneMatrix_vertical_temp_horizontal_original, order='C'):
                if i >= max:
                    max = i
                if i <= min:
                    min = i
            # print("max min: ", max, min)
            delta = int(max) - int(min)
            # print("delta: ", delta)
            resultMatrix = pixelChanneMatrix_vertical_temp_horizontal_original
            loopNumber = 0
            # print("***************************DistilingMatrix Start*************************************")
            while delta > 2:
                loopNumber += 1
                delta, resultMatrix = DistilingMatrix(channelName, loopNumber, delta, resultMatrix)
                # print("delta: ", delta)
                '''
                    print result(last absolute ) matrix
                '''
                # print("result(last absolute )Matrix: \n", resultMatrix)
                # print("loop", loopNumber, "over! \n")
                # print("@" * 100)
                if delta <= 2:
                    # print("\n flagNUmer: ", loopNumber)
                    b_reconstructMatrix = reconstructMatrix(channelName, loopNumber)
                    if horizontal_flag:
                        restore_pixel_matrix_horizontal_temp = b_reconstructMatrix
                        horizontal_flag = False
                    else:
                        restore_pixel_matrix_horizontal_temp = restorePixelMatrixInHorizontal(
                            restore_pixel_matrix_horizontal_temp, b_reconstructMatrix)
                    break
            # break
            count_width += 1
        # print("count_width: ", count_width)  # count_width should be 45
        if vertical_flag:
            restore_pixel_matrix = restore_pixel_matrix_horizontal_temp
            vertical_flag = False
        else:
            restore_pixel_matrix = restorePixelMatrixInVertical(restore_pixel_matrix,
                                                                restore_pixel_matrix_horizontal_temp)
        count_length += 1

    print("\n count_length: ", count_length)  # count_length should be 80
    print(restore_pixel_matrix, '\n', restore_pixel_matrix.shape)
    global channelbuffer
    channelbuffer.update({channelName + "_restore": restore_pixel_matrix})
    # np.savetxt("./buffer/" + channel + "_restore.txt", restore_pixel_matrix, fmt='%d', delimiter=',')


class myThread(threading.Thread):
    def __init__(self, threadID, matrix, channelName):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.matrix = matrix
        self.channelName = channelName

    def run(self):
        print("开启线程：" + self.threadID)
        # 获取锁，用于线程同步
        #threadLock.acquire()
        DistilingProgram(self.matrix, self.channelName)
        print("退出线程：" + self.threadID)
        #threadLock.release()


if __name__ == '__main__':
    # tic = time.time()
    # Image_to_array_file()
    # toc = time.time()
    # print(f"used {toc - tic} s")

    r_matrix, g_matrix, b_matrix, = Image_to_array_file()
    tic = time.time()

    #threadLock = threading.Lock()
    threads = []

    # 创建3个线程
    thread1 = myThread("1", r_matrix, "r")
    thread2 = myThread("2", g_matrix, "g")
    thread3 = myThread("3", b_matrix, "b")

    # 开启新线程
    thread1.start()
    thread2.start()
    thread3.start()

    # 添加线程到线程列表
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)

    # 等待所有线程完成
    for t in threads:
        t.join()
    print("退出主线程")

    # DistilingProgram(r_matrix, "r")
    # DistilingProgram(g_matrix, "g")
    # DistilingProgram(b_matrix, "b")
    toc = time.time()
    print(f"used {toc - tic} s")
    ImageToMatrix()

