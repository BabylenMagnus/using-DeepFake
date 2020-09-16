import yaml
import numpy as np
import torch
from model import Generator, KPDetector
from torch.nn.parallel.data_parallel import DataParallel
from scipy.spatial import ConvexHull
import cv2
from create_video import scale_image


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def load_checkpoints():
    """
    Загружает модели, generator и kp_detector по модели и сохранению
    AliaksandrSiarohin/first-order-model
    """

    with open('data/vox-256.yaml') as f:
        config = yaml.load(f)

    generator = Generator(**config['model_params']['generator_params'],
                          **config['model_params']['common_params'])
    generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.cuda()

    checkpoint = torch.load('data/vox-cpk.pth.tar')

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallel(generator)
    kp_detector = DataParallel(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(image, video, generator, kp_detector, cascade, video_path,
                   coord, long, relative=True, adapt_movement_scale=True):
    with torch.no_grad():

        source = torch.tensor(image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.cuda()
        last_frame = None

        fps = video.get(cv2.CAP_PROP_FPS)
        codec = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter(video_path, codec, fps, (256, 256))

        kp_source = kp_detector(source)
        ret, frame = video.read()
        image = scale_image(frame, coord, long, frame.shape[0])
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (256, 256))
        image = image / 255

        driving = torch.tensor(np.array(image)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

        kp_driving_initial = kp_detector(driving)

        while True:

            frame, last_frame = scale_image(frame, coord, long, frame.shape[0])
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256, 256))
            frame = frame / 255
            driving = torch.tensor(np.array(frame)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

            driving = driving.cuda()
            kp_driving = kp_detector(driving)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            predict = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predict = np.transpose(predict['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            predict = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
            predict = np.uint8(predict * 255)
            out.write(predict)

            ret, frame = video.read()
            if not ret:
                break

    out.release()
