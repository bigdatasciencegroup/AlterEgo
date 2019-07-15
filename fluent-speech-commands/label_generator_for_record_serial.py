"""
Ran using Python 3.6.8

-------Generates labels for thw word_map in record_serial.py--------
total phrases = 275 + 'FINISHED FINISHED FINISHED FINISHED FINISHED'

1 recording session of ~20 minutes
"""


def generate_labels(total_phrases):
    import random

    #total_phrases = 248

    def chunkIt(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out


    total = list(range(total_phrases))
    all_labels = []

    for item in total:
        if item not in all_labels:
            all_labels.append(item)

    random.shuffle(all_labels)

    all_chunks = chunkIt(all_labels, 1)  # 1 session

    print('ALL CHUNKS:')
    for item in all_chunks:
        print(item)

    return all_chunks[0]

generate_labels(275)

'''
OUTPUT (copied from terminal):

ALL CHUNKS:
[174, 150, 106, 5, 116, 114, 247, 184, 39, 26, 98, 72, 243, 168, 43, 244, 201, 187, 181, 229, 65, 179, 4, 105, 231, 194, 192, 154, 21, 103, 89, 145, 153, 246, 156, 173, 135, 6, 80, 228, 208, 15, 22, 12, 9, 63, 175, 139, 163, 210, 149, 120, 52, 226, 142, 87, 54, 164, 78, 7, 199, 108, 220, 20, 10, 238, 14, 0, 58, 95, 240, 16, 189, 221, 112, 191, 2, 242, 209, 161, 178, 157, 96, 211, 172, 41, 101, 77, 83, 104, 29, 147, 122, 88, 51, 37, 60, 148, 42, 207, 180, 223, 33, 17, 185, 53, 234, 196, 23, 55, 73, 205, 86, 115, 237, 130, 182, 25, 117, 171, 24, 128, 79, 18, 38, 158, 70, 188, 127, 61, 143, 245, 97, 82, 100, 19, 44, 133, 109, 1, 102, 111, 144, 215, 57, 224, 40, 176, 230, 113, 170, 125, 76, 162, 140, 190, 94, 126, 218, 71, 195, 118, 50, 212, 204, 134, 132, 136, 62, 107, 34, 169, 222, 214, 177, 129, 74, 225, 198, 206, 11, 239, 3, 92, 59, 193, 36, 183, 66, 49, 197, 227, 151, 233, 216, 166, 186, 56, 69, 165, 121, 84, 32, 93, 219, 67, 217, 27, 202, 203, 48, 236, 31, 124, 131, 167, 152, 35, 81, 119, 64, 46, 30, 146, 13, 123, 232, 91, 200, 155, 110, 213, 8, 85, 141, 241, 138, 47, 99, 45, 90, 137, 159, 68, 75, 28, 160, 235]

'''