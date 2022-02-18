#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
import mediapipe as mp

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--model_selection", type=int, default=0)
    parser.add_argument(
        "--min_detection_confidence",
        help='min_detection_confidence',
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--sface_model",
        type=str,
        default='model/face_recognition_sface_2021dec.onnx',
    )
    parser.add_argument(
        '--sface_input_shape',
        type=str,
        default="112,112",
        help="Specify an input shape for inference.",
    )
    parser.add_argument("--sface_score_th", type=float, default=0.25)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_selection = args.model_selection
    min_detection_confidence = args.min_detection_confidence

    sface_model = args.sface_model
    sface_input_shape = tuple(map(int, args.sface_input_shape.split(',')))
    sface_score_th = args.sface_score_th

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )

    sface = onnxruntime.InferenceSession(
        sface_model,
        providers=['CPUExecutionProvider'],
    )
    sface_input_name = sface.get_inputs()[0].name
    feature_vectors = None

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_detection.process(image)
        scores, bboxes, keypoints_list = postprocess(image, results.detections)

        # 顔識別 ###############################################################
        face_images = crop_face_images(debug_image, bboxes, keypoints_list)
        face_ids = []
        for _, face_image in enumerate(face_images):
            # 前処理
            input_image = cv.resize(
                face_image,
                dsize=(sface_input_shape[1], sface_input_shape[0]),
            )
            input_image = input_image.transpose(2, 0, 1)
            input_image = input_image.astype('float32')
            input_image = np.expand_dims(input_image, axis=0)

            # 推論実施
            result = sface.run(
                None,
                {sface_input_name: input_image},
            )
            result = np.array(result[0][0])

            # 初回推論時のデータ登録
            if feature_vectors is None:
                feature_vectors = copy.deepcopy(np.array([result]))

            # COS類似度計算
            cos_results = cos_similarity(result, feature_vectors)
            max_index = np.argmax(cos_results)
            max_value = cos_results[max_index]

            if max_value < sface_score_th:
                # スコア閾値以下であれば特徴ベクトルリストに追加
                feature_vectors = np.vstack([
                    feature_vectors,
                    result,
                ])
            else:
                # スコア閾値以上であれば顔認証のIDを追加
                face_ids.append(max_index)

        # 描画 ################################################################
        debug_image = draw_detection(
            debug_image,
            scores,
            bboxes,
            keypoints_list,
            face_ids,
            display_fps,
        )

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Face Detection & SFace Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def postprocess(image, face_detection_result):
    score_list = []
    bbox_list = []
    keypoints_list = []

    if face_detection_result is None:
        return score_list, bbox_list, keypoints_list

    image_width, image_height = image.shape[1], image.shape[0]

    for detection in face_detection_result:
        # スコア
        score = detection.score[0]
        score_list.append(score)

        # バウンディングボックス
        bbox = detection.location_data.relative_bounding_box
        xmin = int(bbox.xmin * image_width)
        ymin = int(bbox.ymin * image_height)
        xmax = int(xmin + (bbox.width * image_width))
        ymax = int(ymin + (bbox.height * image_height))

        bbox_list.append([xmin, ymin, xmax, ymax])

        # キーポイント：右目
        keypoint0 = detection.location_data.relative_keypoints[0]
        right_eye_x = int(keypoint0.x * image_width)
        right_eye_y = int(keypoint0.y * image_height)

        # キーポイント：左目
        keypoint1 = detection.location_data.relative_keypoints[1]
        left_eye_x = int(keypoint1.x * image_width)
        left_eye_y = int(keypoint1.y * image_height)

        # キーポイント：鼻
        keypoint2 = detection.location_data.relative_keypoints[2]
        nose_x = int(keypoint2.x * image_width)
        nose_y = int(keypoint2.y * image_height)

        # キーポイント：口
        keypoint3 = detection.location_data.relative_keypoints[3]
        mouth_x = int(keypoint3.x * image_width)
        mouth_y = int(keypoint3.y * image_height)

        # キーポイント：右耳
        keypoint4 = detection.location_data.relative_keypoints[4]
        right_ear_x = int(keypoint4.x * image_width)
        right_ear_y = int(keypoint4.y * image_height)

        # キーポイント：左耳
        keypoint5 = detection.location_data.relative_keypoints[5]
        left_ear_x = int(keypoint5.x * image_width)
        left_ear_y = int(keypoint5.y * image_height)

        keypoints_list.append([
            [right_eye_x, right_eye_y],
            [left_eye_x, left_eye_y],
            [nose_x, nose_y],
            [mouth_x, mouth_y],
            [right_ear_x, right_ear_y],
            [left_ear_x, left_ear_y],
        ])

    return score_list, bbox_list, keypoints_list


def image_rotate(image, angle, scale=1.0):
    image_width, image_height = image.shape[1], image.shape[0]
    center = (int(image_width / 2), int(image_height / 2))

    rotation_mat_2d = cv.getRotationMatrix2D(center, angle, scale)

    result_image = cv.warpAffine(
        image,
        rotation_mat_2d,
        (image_width, image_height),
        flags=cv.INTER_CUBIC,
    )

    return result_image


def crop_face_images(image, bboxes, keypoints_list):
    image_height, image_width = image.shape[0], image.shape[1]

    face_image_list = []
    for bbox, keypoints in zip(bboxes, keypoints_list):
        xmin = int(np.clip(bbox[0], 0, image_width - 1))
        ymin = int(np.clip(bbox[1], 0, image_height - 1))
        xmax = int(np.clip(bbox[2], 0, image_width - 1))
        ymax = int(np.clip(bbox[3], 0, image_height - 1))
        face_image = copy.deepcopy(image[ymin:ymax, xmin:xmax])

        right_eye = keypoints[0]
        left_eye = keypoints[1]
        mouth = keypoints[3]

        a = np.array([((right_eye[0] + left_eye[0]) / 2),
                      ((right_eye[1] + left_eye[1]) / 2)])
        b = np.array([mouth[0], mouth[1]])
        vec = b - a
        angle = math.degrees(np.arctan2(vec[0], vec[1]))

        face_image = image_rotate(face_image, -angle)
        face_image_list.append(face_image)

    return face_image_list


def cos_similarity(X, Y):
    Y = Y.T

    # (128,) x (n, 128) = (n,)
    result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))

    return result


def draw_detection(image, scores, bboxes, keypoints_list, face_ids, fps):
    for score, bbox, keypoints, face_id in zip(scores, bboxes, keypoints_list,
                                               face_ids):
        # バウンディングボックス
        cv.rectangle(
            image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2,
        )

        # 顔認証ID
        cv.putText(
            image,
            'Face ID:' + str(face_id),
            (bbox[0], bbox[1] - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        for keypoint in keypoints:
            cv.circle(image, (keypoint[0], keypoint[1]), 5, (0, 255, 0), 2)

    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 255, 0), 2, cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
