import numpy as np
import scipy as sp
from scipy import fftpack
import math
from sys import argv
import warnings
from scipy.misc import imread

'''
    @autor Deniz Şen
    @title Baseline JPEG Encoder for Grayscale Images
    @date 14.01.2020
'''

LOW_COMPRESSION_QT =  [3,   2,   2,   3 ,  4  , 6 ,  8 , 10 ,
  2 ,  2 ,  2 ,  3  , 4 ,  9 , 10  , 9 ,
  2  , 2 ,  3 ,  4 ,  6 ,  9 , 11 ,  9 ,
   2 ,  3 ,  4 ,  5 ,  8  ,14 , 13 ,10 ,
  3  , 4  , 6 ,  9,  11 , 17,  16,  12 ,
  4 ,  6  , 9  ,10 , 13,  17 , 18 , 15 ,
   8  ,10  ,12,  14 , 16 , 19,  19 , 16 ,
  12 , 15 , 15  ,16 , 18,  16 , 16,  16 ]

MIDDLE_COMPRESSION_QT = [     8, 6, 5, 8,12,20,26,31
,6, 6, 7,10,13,29,30,28
, 7, 7, 8,12,20,29,35,28
, 7, 9,11,15,26,44,40,31
,9,11,19,28,34,55,52,39
,12,18,28,32,41,52,57,46
,25,32,39,44,52,61,60,51
,36,46,48,49,56,50,52,50
]


HIGH_COMPRESSION_QT = [16, 11, 10, 16, 24, 40, 51, 61,
                       12, 12, 14, 19, 26, 58, 60, 55,
                       14, 13, 16, 24, 40, 57, 69, 56,
                       14, 17, 22, 29, 51, 87, 80, 62,
                       18, 22, 37, 56, 68, 109, 103, 77,
                       24, 35, 55, 64, 81, 104, 113, 92,
                       49, 64, 78, 87, 103, 121, 120, 101,
                       72, 92, 95, 98, 112, 100, 103, 99]

ORDER = np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
                           49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])

DC_SIZE_TO_CODE = [
    [1,1,0],              #0 EOB
    [1,0,1],            #1
    [0,1,1],            #2
    [0,1,0],            #3
    [0,0,0],            #4
    [0,0,1],            #5
    [1,0,0],            #6
    [1,1,1,0],        #7
    [1,1,1,1,0],      #8
    [1,1,1,1,1,0],    #9
    [1,1,1,1,1,1,0],  #10 0A
    [1,1,1,1,1,1,1,0] #11 0B
]

AC_SIZE_TO_CODE = {
'01':[0,0],


'02':[0,1],


'03':[1,0,0],


'11':[1,0,1,0],


'04':[1,0,1,1],


'00':[1,1,0,0],


'05':[1,1,0,1,0],


'21':[1,1,0,1,1],


'12':[1,1,1,0,0],


'31':[1,1,1,0,1,0],


'41':[1,1,1,0,1,1],


'51':[1,1,1,1,0,0,0],


'06':[1,1,1,1,0,0,1],


'13':[1,1,1,1,0,1,0],


'61':[1,1,1,1,0,1,1],


'22':[1,1,1,1,1,0,0,0],


'71':[1,1,1,1,1,0,0,1],


'81':[1,1,1,1,1,0,1,0,0],


'14':[1,1,1,1,1,0,1,0,1],


'32':[1,1,1,1,1,0,1,1,0],


'91':[1,1,1,1,1,0,1,1,1],


'A1':[1,1,1,1,1,1,0,0,0],


'07':[1,1,1,1,1,1,0,0,1],


'15':[1,1,1,1,1,1,0,1,0,0],


'B1':[1,1,1,1,1,1,0,1,0,1],


'42':[1,1,1,1,1,1,0,1,1,0],


'23':[1,1,1,1,1,1,0,1,1,1],


'C1':[1,1,1,1,1,1,1,0,0,0],


'52':[1,1,1,1,1,1,1,0,0,1],


'D1':[1,1,1,1,1,1,1,0,1,0],


'E1':[1,1,1,1,1,1,1,0,1,1,0],


'33':[1,1,1,1,1,1,1,0,1,1,1],


'16':[1,1,1,1,1,1,1,1,0,0,0],


'62':[1,1,1,1,1,1,1,1,0,0,1,0],


'F0':[1,1,1,1,1,1,1,1,0,0,1,1],


'24':[1,1,1,1,1,1,1,1,0,1,0,0],


'72':[1,1,1,1,1,1,1,1,0,1,0,1],


'82':[1,1,1,1,1,1,1,1,0,1,1,0,0],


'F1':[1,1,1,1,1,1,1,1,0,1,1,0,1],


'25':[1,1,1,1,1,1,1,1,0,1,1,1,0,0],


'43':[1,1,1,1,1,1,1,1,0,1,1,1,0,1],


'34':[1,1,1,1,1,1,1,1,0,1,1,1,1,0],


'53':[1,1,1,1,1,1,1,1,0,1,1,1,1,1],


'92':[1,1,1,1,1,1,1,1,1,0,0,0,0,0],


'A2':[1,1,1,1,1,1,1,1,1,0,0,0,0,1],


'B2':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,0],


'63':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,1],


'73':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0],


'C2':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1],


'35':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0],


'44':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1],


'27':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0],


'93':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1],


'A3':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,0],


'B3':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1],


'36':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,0],


'17':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,1],


'54':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,0],


'64':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1],


'74':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0],


'C3':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1],


'D2':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,0],


'E2':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1],


'08':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0],


'26':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1],


'83':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0],


'09':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1],


'0A':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0],


'18':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,1],


'19':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0],


'84':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,1],


'94':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0],


'45':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,1],


'46':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0],


'A4':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1],


'B4':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0],


'56':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,1],


'D3':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],


'55':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1],


'28':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,0],


'1A':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,1],


'F2':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0],


'E3':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1],


'F3':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0],


'C4':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,1],


'D4':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0],


'E4':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1],


'F4':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,0],


'65':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1],


'75':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0],


'85':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1],


'95':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0],


'A5':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1],


'B5':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0],


'C5':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1],


'D5':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0],


'E5':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1],


'F5':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0],


'66':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],


'76':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],


'86':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1],


'96':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0],


'A6':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1],


'B6':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0],


'C6':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1],


'D6':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0],


'E6':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1],


'F6':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0],


'37':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1],


'47':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0],


'57':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1],


'67':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0],


'77':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1],


'87':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0],


'97':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1],


'A7':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0],


'B7':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1],


'C7':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0],


'D7':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1],


'E7':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0],


'F7':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],


'38':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,0],


'48':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1],


'58':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0],


'68':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1],


'78':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0],


'88':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1],


'98':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0],


'A8':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1],


'B8':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0],


'C8':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1],


'D8':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],


'E8':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1],


'F8':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0],


'29':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1],


'39':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0],


'49':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1],


'59':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0],


'69':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1],


'79':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0],


'89':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1],


'99':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0],


'A9':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1],


'B9':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0],


'C9':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1],


'D9':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0],


'E9':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1],


'F9':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],


'2A':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1],


'3A':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0],


'4A':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],


'5A':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0],


'6A':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1],


'7A':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0],


'8A':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1],


'9A':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],


'AA':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1],


'BA':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0],


'CA':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1],


'DA':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],


'EA':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],


'FA':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
}


def hex_sum(lst):
    '''
    Method to add hexadecimal numbers
    :param lst: a list of hexadecimal numbers
    :return: decimal value of their sum
    '''
    sum = 0
    for i in range(len(lst)):
        sum += (lst[i]) * (16 ** (2 * i))
    return sum

def quantization(used_qm, file_bmp):
    '''
    Quantization on the blocks
    :param used_qm: the quantization matrrix to be used inside the method
    :param file_bmp: path of the bitmap file
    :return: the quantized image, its dimensions and the ac and dc coefficients
    '''

    block_size = 8
    #try:
    y = imread(file_bmp, mode="YCbCr")[: ,: , 0]
    #except Exception:
    #    print("Error while reading th file!")
    #    exit(1)

    y = y.astype('int')
    y = np.subtract(y, 128)
    ac_coefs = []
    dc_coefs = []

    height = y.shape[0]
    width = y.shape[1]

    y = y.reshape(height, width)
    x_block_num = width // block_size
    y_block_num = height // block_size

    quantized_image = np.zeros(y.shape)

    # DCT and Quantization in blocks
    for i in range(y_block_num):
        for j in range(x_block_num):
            #take the block
            current_block = y[i * (block_size):(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            block_dct = fftpack.dct(fftpack.dct(current_block.T, norm='ortho').T, norm='ortho')
            quantized_block = np.zeros(block_dct.shape)
            #quantize the values
            for k in range(block_size):
                for l in range(block_size):
                    m = block_dct[k][l]
                    q = np.rint(block_dct[k][l] / used_qm[k *8 + l])
                    quantized_block[k][l] = q

            quantized_image[i * (block_size):(i + 1) * block_size,
            j * block_size:(j + 1) * block_size] = quantized_block

    remaining_x = width - (x_block_num * block_size)
    remaining_y = height - (y_block_num * block_size)

    #add rest of the image
    quantized_image[(y_block_num * block_size):height, :] = y[(y_block_num * block_size):height, :]
    quantized_image[:, (x_block_num * block_size):width] = y[:, (x_block_num * block_size):width]

    return quantized_image, (height, width), ac_coefs, dc_coefs

def add_00_after_ff(bin_str):
    '''
    Add 00 after FF bytes to prevent confusion with the headers
    :param bin_str: binary string
    :return: binary string with FF00 blocks
    '''
    i = 0
    try:
        while True:
            if 8*i >= len(bin_str):
                break
            current_str = bin_str[8 * i:(i + 1) * 8]
            if current_str == '11111111':
                prefix = bin_str[0:(i + 1) * 8]
                postfix =  bin_str[(i + 1) * 8:len(bin_str)]
                bin_str = prefix + "00000000" + postfix
                i = i + 1
            i = i + 1
    except e:
        print()
    return bin_str

def ones_complement(s):
    '''
    Ones complement of the input binary string
    :param s: binary string
    :return: ones complement of the input
    '''
    a = ""
    for i in s:
        if i == '0':
            a = a + "1"
        else:
            a = a + "0"
    return a

def list_to_string(l):
    '''
    Convert a list of values into a string
    :param l: list of values
    :return: string of the list
    '''
    a = ""
    for i in l:
        a = a + str(i)
    return a

def dc_encode(q_image):
    '''
    Encode the DC coefficients with DPCM
    :param q_image: the quantized image
    :return: list of encoded dc coefficients
    '''
    dc_coefficients = q_image[0::8, 0::8]
    dc_coefficients = np.reshape(dc_coefficients, dc_coefficients.shape[0]*dc_coefficients.shape[1])
    #DPCM
    for i in np.arange(len(dc_coefficients)-1, 0, -1):
        dc_coefficients[i] = dc_coefficients[i] - dc_coefficients[i-1]

    codes = []
    for i in range(len(dc_coefficients)):
        v = dc_coefficients[i]
        if v == 0:
            size = 0
            value = ""
        else:
            value = str(bin(int(np.abs(dc_coefficients[i])))[2:]) if v >= 0 else ones_complement(str(bin(int(np.abs(dc_coefficients[i])))[2:]))
            size = len(value)
        code = list_to_string(DC_SIZE_TO_CODE[size])
        codes.append(code + value)

    return codes

def zigzag(array):
    '''
    Remake the list according t the zigzag order
    :param array: list of values
    :return: zigzag ordered list
    '''
    n = []
    for i in ORDER:
        n.append(array[i])
    return n

def ac_encode(q_image):
    '''
    Encode the AC coefficients with the Huffman Table and Run length Encoding
    :param q_image: Quantized image
    :return: List of encoded AC coefficients of each block
    '''
    ac_coefficients = []
    x_block_num = q_image.shape[1] // 8
    y_block_num = q_image.shape[0] // 8
    ac_coefficients = []
    #for each block
    for q in range(y_block_num):
        for w in range(x_block_num):
            bin_str = ""
            current_block = np.reshape(q_image[q * (8):(q + 1) * 8, w * 8:(w + 1) * 8], 64)

            current_block = zigzag(current_block)
            current_block = current_block[1:]
            for m in range(len(current_block)):
                current_block[m] = int(current_block[m])
            
            i = 0
            while i < len(current_block):
                run = 0
                # check if rest of current_block are all zero. If so, just write EOB and return
                j = i
                cont = False
                while current_block[j] == 0:
                    if j==len(current_block) - 1:
                        bin_str += list_to_string(AC_SIZE_TO_CODE['00'])
                        cont = True
                        break
                    j = j + 1
                if cont:
                    break
                # stop after 15 zero-length and encode '00'(ZRL)
                while current_block[i]==0 and i < len(current_block) - 1 and run != 15:
                    run = run + 1
                    i = i + 1

                value = current_block[i]

                #stop if the end is reached
                if value == 0 and run != 15:
                    break

                #0 is taken 0 length in bit_length()
                size = int(value).bit_length()
                #encode the zero run-length and the size of the non-zero coefficient
                size = str.upper(str(hex(run))[2:]) + str.upper(str(hex(size))[2:])
                bin_str += list_to_string(AC_SIZE_TO_CODE[size])

                #encode the value of the non-zero coefficient
                if value == 0:
                    code = ""
                else:
                    code = str(bin(int(np.abs(value)))[2:]) if value >= 0 else ones_complement(str(bin(int(np.abs(value)))[2:]))
                bin_str += list_to_string(code)

                i = i + 1
            ac_coefficients.append(bin_str)
    return ac_coefficients

def encode_quantized_image(q_image):
    '''
    Add the encoded AC and DC coefficients of the block.
    Then, add 00 after each FF byte
    :param q_image: Quantized image
    :return: Binary string, encoding the quantized image
    '''
    dc = []
    dc_coefs = dc_encode(q_image)
    ac_coefs = ac_encode(q_image)
    bin_str = ""

    for i in range(len(dc_coefs)):
        bin_str += dc_coefs[i] + ac_coefs[i]

    bin_str  = add_00_after_ff(bin_str)

    return bin_str


def encode_quantization_table(qt):
    '''
    Encode the quantization table
    :param qt: the quantization table
    :return: binary string encoding the quantization table
    '''
    bin_str = ""
    for i in ORDER:
            bin_str += str(bin(qt[i]))[2:].zfill(8)

    return bin_str

def binary_to_bytes(bin_str, add=1):
    '''
    Convert binary string to bytes
    Pad 0s to complete bytes
    :param bin_str: binary string
    :param add: the position of the padding
    :return: bytes of the binary string
    '''
    #add 0s
    while True:
        if len(bin_str) % 8 == 0:
            break
        if add == 1:
            bin_str += '0'
        elif add == -1:
            bin_str = '0' + bin_str
        else:
            break

    num = len(bin_str) // 8
    ret = np.zeros([num], dtype= int)
    for i in range(num):
        ret[i] = int(bin_str[i*8:(i+1)*8], 2)
    ret = ret.tolist()
    ret = bytes(ret)
    return ret

def int_to_bytes(integer, unsigned=True, length=16):
    '''
    Convert integer to bytes
    :param integer: The integer
    :param unsigned:
    :param length: LEngth of the bytes value
    :return: bytes version of the integer
    '''
    if not unsigned or integer < 0:
        return binary_to_bytes("1" + str(bin(integer))[3:].zfill(length - 1))
    else:
        return binary_to_bytes(str(bin(integer))[2:].zfill(length))

def hex_to_bin(hex_str):
    '''
    Convert HEX string to Binary string
    :param hex_str: HEX string
    :return: Binary Representation of the HEX string
    '''
    n = len(hex_str) // 2
    bin_str = ""
    for i in range(n):
        bin_str += bin(int(hex_str[i*2:(i+1)*2], 16))[2:].zfill(8)
    return bin_str

def hex_to_bytes(hex_str):
    '''
    Convert HEX string to bytes
    :param hex_str: HEX string
    :return: Bytes version of the string
    '''
    return binary_to_bytes(hex_to_bin(hex_str))



def encode_huffman_table(huff_table, sorted):
    '''
    Unused Huffman encoding
    :param huff_table:
    :param sorted:
    :return:
    '''
    ht = {}
    s = sorted[0]
    for i in np.arange(len(s)-1, -1, -1):
        ht[s[i]] =  huff_table[s[i]]

    huff_table = ht
    bucket = []

    for i in range(16):
        bucket.append([])

    for k in huff_table:
        (bucket[len(huff_table[k])-1]).append((k, huff_table[k]))

    bin_str = ""
    for i in bucket:
        bin_str += str(bin(int(len(i)))[2:].zfill(8))
    for i in bucket:
        for j in i:
            if j[0] < 0:
                bin_str += '1' + str(bin(int(j[0]))[3:].zfill(7))
            else:
                bin_str += str(bin(int(j[0]))[2:].zfill(8))

    return bin_str


def main(image, quantization_level):
    sp.seterr(all="ignore")
    used_qm = MIDDLE_COMPRESSION_QT
    if quantization_level == 1:
        used_qm = LOW_COMPRESSION_QT
    elif quantization_level == 2:
        used_qm = MIDDLE_COMPRESSION_QT
    elif quantization_level == 3:
        used_qm = HIGH_COMPRESSION_QT
    else:
        print("Quantization level can only be between 1 and 3")
        exit(1)
    
    quantized_image, shape, ac_coefs, dc_coefs = quantization(used_qm, image)

    encoded_qt = encode_quantization_table(used_qm)

    output_file = image.split("/")[-1].split(".")[0] + ".jpg"
    jpg_fd = open(output_file, "wb")

    #write SOI(Start of Image)
    jpg_fd.write(hex_to_bytes("FFD8"))

    #APP0 segment
    jpg_fd.write(hex_to_bytes('FFE000104A46494600010100000100010000'))

    # write Define Quantization Table(DQT)
    jpg_fd.write(hex_to_bytes("FFDB"))

    # write the size of DQT
    b = binary_to_bytes(encoded_qt)
    jpg_fd.write(int_to_bytes(len(b) + 3))
    print(int_to_bytes(len(b) + 3))

    # mode of the DQT: 00h for Y
    jpg_fd.write(hex_to_bytes("00"))

    # write encoded quantization table
    jpg_fd.write(binary_to_bytes(encoded_qt))

    jpg_fd.write(hex_to_bytes('FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))

    # write Start of Frame (SOF)
    # write the length of the segment (fixed)
    jpg_fd.write(hex_to_bytes("FFC0000B08"))

    # write height of the image
    jpg_fd.write(int_to_bytes(shape[0]))

    # write width of the image
    jpg_fd.write(int_to_bytes(shape[1]))

    # write number of components and component ids
    # Note: probably fixed
    jpg_fd.write(hex_to_bytes("01011100"))

    #SOS
    jpg_fd.write(hex_to_bytes("FFDA0008010100003F00"))
    a = encode_quantized_image(quantized_image)
    jpg_fd.write(binary_to_bytes(encode_quantized_image(quantized_image), add=1))

    #EOI
    jpg_fd.write(hex_to_bytes("FFD9"))
    jpg_fd.close()


if __name__=='__main__':
    main(image=argv[1], quantization_level=int(argv[2]))

