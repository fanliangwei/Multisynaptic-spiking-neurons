import cv2
import numpy as np
import torch
import torch.nn as nn

class OpticFlow(object):
    def __init__(self, mode='pcaflow', is_color = True):
        self.mode = mode
        self.is_color = is_color

        self.setmode(self.mode)

    def setmode(self, mode='pcaflow'):
        self.mode = mode
        if self.mode == 'deepflow':
            self.inst = cv2.optflow.createOptFlow_DeepFlow()
        elif self.mode == 'pcaflow':
            # self.inst = cv2.optflow.createOptFlow_PCAFlow()
            self.inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        elif self.mode == 'disflow':
            self.inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.inst.setUseSpatialPropagation(True)
        else:
            self.inst = None

    @staticmethod
    def video_BGR2GRAY(video):
        video = video.permute(0, 1, 3, 4, 2)
        gray = []
        for batch in video:
            temp = []
            for frame in batch:
                temp.append([cv2.cvtColor(frame.detach().numpy(), cv2.COLOR_BGR2GRAY)])
            gray.append(temp)
        return torch.tensor(gray).float()


            
    @staticmethod
    def predict_next_tensor_video(img1,flows):
        flows = flows.permute(1,0,2,3,4)

        img2 = []
        for flow in flows:
            img2.append(OpticFlow.predict_next_tensor(img1,flow).numpy())
        return torch.tensor(img2).permute(1,0,2,3,4)

    @staticmethod
    def predict_next(img1,flow):
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        w,h = int(img1.shape[1]), int(img1.shape[0])
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.float32(np.dstack([x_coords, y_coords]))
        pixel_map = coords + flow
        new_frame = cv2.remap(img1, pixel_map,None,cv2.INTER_LINEAR)
        return new_frame

    @staticmethod
    def predict_next_tensor(img1,flow):
        img1 = img1.permute(0,2,3,1)*255
        flow = flow.permute(0,2,3,1)

        img2 = []
        for item1, item2 in zip(img1, flow):
            img2.append([OpticFlow.predict_next(item1.detach().numpy().astype('uint8'),item2.detach().numpy())])
        img2 = torch.tensor(img2).float()/255
        return img2

    def getflow_tensor_video(self,img1,video):
        img1 = img1.permute(0,2,3,1)*255
        video = video.permute(0,1,3,4,2)*255

        flows = []
        for batch in range(video.size(0)):
            temp = []
            for frame in video[batch]:
            # print(item1.size(),item2.size())
                temp.append(self.getflow(img1[batch].detach().numpy().astype('uint8'),frame.detach().numpy().astype('uint8')))
            flows.append(temp)
        flows = torch.tensor(flows).float()
        return flows.permute(0, 1, 4, 2, 3)

    def getflow_tensor(self,img1,img2):
        img1 = img1.permute(0,2,3,1)*255
        img2 = img2.permute(0,2,3,1)*255

        flows = []
        for item1, item2 in zip(img1,img2):
            # print(item1.size(),item2.size())
            flows.append(self.getflow(item1.detach().numpy().astype('uint8'),item2.detach().numpy().astype('uint8')))
        flows = torch.tensor(flows).float()
        return flows.permute(0,3,1,2)

    def getflow(self,img1, img2):
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        # print(np.shape(img1),np.shape(img2),type(img1[0][0]),type(img2[0][0]))
        if self.mode == 'franeback':
            flow = cv2.calcOpticalFlowFarneback(prev=img1, next=img2, flow=None, pyr_scale=0.5, levels=5,
                                            winsize=15,
                                            iterations=3, poly_n=3, poly_sigma=1.2,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        elif self.mode == 'simpleflow':
            flow = cv2.optflow.calcOpticalFlowSF(img1, img2, 2, 2, 4)
        elif self.mode == 'denseflow':
            flow = cv2.optflow.calcOpticalFlowSparseToDense(img1, img2)
        else:
            assert self.inst != None, 'please input right mode name!'
            flow = self.inst.calc(img1,img2, None)
        return flow

    @staticmethod
    def show_flow_hsv(flow, show_style=1):
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])#将直角坐标系光流场转成极坐标系
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
        #光流可视化的颜色模式
        if show_style == 1:
            hsv[..., 0] = ang * 180 / np.pi / 2 #angle弧度转角度
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#magnitude归到0～255之间
        elif show_type == 2:
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = 255
        #hsv转bgr
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr


if __name__ == '__main__':
    of = OpticFlow()

    # BGR2GRAY video
    # video = torch.randn([1,8,3,256,256])
    # print(of.video_BGR2GRAY(video).size())

    # cal flow (255, 255, 3) numpy
    img1 = cv2.imread('test/1.jpg')
    img2 = cv2.imread('test/2.jpg')
    flow = of.getflow(img1, img2)
    print(np.min(flow), np.max(flow))
    a = of.show_flow_hsv(flow)
    cv2.imshow('',a)
    cv2.waitKey(0)