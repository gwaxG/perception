import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import rospy, socket, pickle
from std_msgs.msg import Float32MultiArray
from collections import defaultdict

class AttributeDict(defaultdict):
    def __init__(self):
        super(AttributeDict, self).__init__(AttributeDict)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

class Detector:
    def __init__(self):
        rospy.init_node('server_listener')
        self.pub = rospy.Publisher('/bbox', Float32MultiArray)
        self.msg = Float32MultiArray()
        self.opt = AttributeDict()
        self.opt.cfg = 'cfg/yolov3-tiny-1cls.cfg'
        self.opt.conf_thres = 0.3
        self.opt.data = 'stairs.data'
        self.opt.device = ''
        self.opt.fourcc = 'mp4v' 
        self.opt.half = False
        self.opt.img_size = 416
        self.opt.nms_thres = 0.5 
        self.opt.output = 'output' 
        self.opt.source = 'images/'
        self.opt.view_img = False
        self.opt.weights = 'weights/best.pt'
        # 1
        self.img_size = (320, 192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        self.out, self.source, self.weights, self.half, self.view_img = self.opt.output, self.opt.source, self.opt.weights, self.opt.half, self.opt.view_img
        self.webcam = self.source == '0' or self.source.startswith('rtsp') or self.source.startswith('http') or self.source.endswith('.txt')

        # Initialize
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete output folder
        os.makedirs(self.out)  # make new output folder

        # Initialize model
        self.model = Darknet(self.opt.cfg, self.img_size)

        # Load weights
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, self.weights)
        # 2
        self.model.to(self.device).eval()
        # ONNX_EXPORT ?
        # Export mode
        if ONNX_EXPORT:
            self.img = torch.zeros((1, 3) + self.img_size)  # (1, 3, 320, 192)
            torch.onnx.export(self.model, img, 'weights/export.onnx', verbose=True)
            return

        # Half precision
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None

        self.view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        # Initialize class which implements __next__

        # Get classes and colors
        self.classes = load_classes(parse_data_cfg(self.opt.data)['names'])
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

        # Run inference
        self.t0 = time.time()
        self.TCP_IP = 'localhost'
        self.TCP_PORT = 5002
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.TCP_IP, self.TCP_PORT))

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: 
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def detect(self, img0):
        '''
        :param img0: read frame
        :return dataset: a list with appropriate information
        '''
        # Pre-processing from "datasets.py"
        # For a reason
        img0 = cv2.flip(img0, 1)  # flip left-right
        
        img = letterbox(img0, new_shape=self.img_size)[0]
        
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # img_path is not used
        # also, img0 is a cv2 image get from cvBridge application

        # Detection itself
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]

        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.nms_thres)

        # im0s == img0
        bboxes = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:

                    bboxes.append([float(it) for it in xyxy])
        return bboxes

    def find_biggest(self, data):
        square = 0
        for i, l in enumerate(data):
            cur = (l[2] - l[0]) * (l[3] - l[1])
            if cur > square:
                ind = i
                square = cur
        return data[ind][:4] if square != 0 else []

    def run(self):
        while True:
            self.s.listen(True)
            conn, addr = self.s.accept()
            length = self.recvall(conn,16)
            stringData = self.recvall(conn, int(length))
            data = np.fromstring(stringData, dtype='uint8')  
            img = cv2.imdecode(data,1)
            res = self.detect(img)
            res = self.find_biggest(res)
            data = pickle.dumps(res,protocol=2)
            conn.send(data)
            # self.msg.data = self.find_biggest(res)
            # self.pub.publish(self.msg)

if __name__ == '__main__':
    Detector().run()
