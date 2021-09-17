import _queue
import argparse
import multiprocessing as mp
import sys
import time
from os import getpid

import cv2

from PaddleOCR.tools.infer import utility
from PaddleOCR.tools.infer.predict_e2e import TextE2E

WK_MSG_INIT_SUCCESS = 'INIT_SUCCESS'
WK_MSG_END = 'END'


def worker(input_q: mp.Queue, output_q: mp.Queue):
    text_detector = TextE2E(utility.parse_args())

    dummy_img = cv2.imread('./PaddleOCR/doc/imgs_en/1_cropped.png')
    for i in range(100):
        tmp = time.time()
        _, _, _ = text_detector(dummy_img)
        print("[DUMMY] Time taken: {}".format(time.time() - tmp))

    output_q.put(WK_MSG_INIT_SUCCESS)

    running = True
    avg_pred_elapsed = 0
    pred_cnt = 0
    while running:
        img = input_q.get()

        if img == WK_MSG_END:
            break

        tmp = time.time()
        _, strs, _ = text_detector(img)
        pred_elapsed = time.time() - tmp
        print("Time taken: {}".format(pred_elapsed))
        avg_pred_elapsed += pred_elapsed
        pred_cnt += 1

        output_q.put(strs)

    avg_pred_elapsed /= pred_cnt
    print("[{}]: avg. pred time: {:.3f}".format(getpid(), avg_pred_elapsed))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-num-i', '--num-images', dest='num_imgs', type=int,
                        default=50, help='Number of images.')

    return parser.parse_args()


def main():
    args = parse_args()

    print("RUN: {} Workers; {} Images...".format(args.num_workers, args.num_imgs))

    print("Creating pool...")
    input_q = mp.Queue()
    output_q = mp.Queue()
    pool = mp.Pool(args.num_workers, worker, (input_q, output_q))

    try:
        for _ in range(args.num_workers):
            msg = output_q.get(timeout=30)
            if msg != WK_MSG_INIT_SUCCESS:
                sys.exit("WORKER_INIT_FAIL")
    except _queue.Empty:
        for _ in range(args.num_workers):
            input_q.put(WK_MSG_END)
        pool.close()
        pool.join()
        input_q.close()
        output_q.close()
        sys.exit("POOL_INIT_TIMEOUT_EXCEEDED")

    print('Pool init successfully')

    imgs = []
    for _ in range(args.num_imgs):
        img = cv2.imread('./PaddleOCR/doc/imgs_en/1_cropped.png')
        imgs.append(img)

    tmp = time.time()
    for i in range(len(imgs)):
        input_q.put(imgs[i])

    for i in range(len(imgs)):
        output_q.get()

    print('Finished!')
    elapsed_time = time.time() - tmp
    print("[INFO] Elapsed prediction time: {:.3f}".format(elapsed_time))

    print('Cleaning resources...')

    for _ in range(args.num_workers):
        input_q.put(WK_MSG_END)
    pool.close()
    pool.join()
    input_q.close()
    output_q.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    main()
