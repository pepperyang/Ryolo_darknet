import cv2
import numpy as np 
import math
import os


out_dir = "./Ryolo_test_out/"
out_txt = './result.txt'
class box():
    def __init__(self):
        self.left_x = 0    
        self.top_y = 0     
        self.width = 0
        self.height = 0
        self.theta = 0
def distance(x1,y1,x2,y2):
	x = (x1-x2)**2
	y = (y1-y2)**2
	distance = math.sqrt(x+y)
	return distance
def draw_line(img,x0,y0,θ):

	if θ == 360:
		θ = 0 
	if θ <= 90:
		θ = θ
	if 90<θ < 180:
		θ = θ-180
	if 180<θ<270:
		θ = θ-180
	else:
		θ = θ-360

	ptStart = (int(x0), int(y0))
	ptEnd = (int(x0+100), int(100*math.tan(math.pi / 180.0 * θ)))
	point_color = (0, 255, 0) # BGR
	thickness = 1
	lineType = 4
	print(ptStart)
	print(ptEnd)
	cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
	return 0

def resize_angle(θ):

	if θ == 360:
		return 0
	if θ == 0:
		return 0
	if θ<90:
		θ = θ
	if 90<θ<180:
		θ = θ-90
		θ = -θ
	if 180<θ<270:
		θ = θ-180
	else:
		θ = 360-θ
		θ = -θ
	return θ

def k(x0,y0,θ):

	if θ==0:
		return x0-100,y0
	if θ==360:
		return x0-100,y0

	if θ<90:
		k = math.tan(math.pi / 180.0 * θ)
		# print(k)
		# exit()
		return x0-50/k,y0+50

	if 90<θ<180:
		k = math.tan(math.pi / 180.0 * (180-θ))
		return x0+50/k,y0+50

	if 180<θ<270:
		k = math.tan(math.pi / 180.0 * (θ-180))
		return x0+50/k,y0-50
	if 270<θ<360:

		k = math.tan(math.pi / 180.0 * (360-θ))
		return x0-50/k,y0-50
	else:
		return x0 - 100, y0

def convert(x1,y1,x2,y2,row,θ): #旋转后的坐标位置在平面坐标上，row为图像的高度
	#任意点P(x1,y1)，绕一个坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)的计算公式：

	x1 = x1
	y1 = row - y1
	x2 = x2
	y2 = row - y2
	x = (x1 - x2)*math.cos(math.pi / 180.0 * θ) - (y1 - y2)*math.sin(math.pi / 180.0 * θ) + x2
	y = (x1 - x2)*math.sin(math.pi / 180.0 * θ) + (y1 - y2)*math.cos(math.pi / 180.0 * θ) + y2
	y = row - y
	return x,y

with open(out_txt) as f:
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)


	lines = f.readlines()
	for i in range(len(lines)):

		line = lines[i].split( )
		print(line)

		if line[0][0] == "C":
			img_path = []
			out_path = []
			img_path.append(line[0])
			out_path.append(out_dir+img_path[0].split("\\")[-1])
			img = cv2.imread(img_path[-1], 1)
		if line[0][0] != "C":
			print(line)

			x_1 = float(line[2])
			y_1 = float(line[4])
			x0 = x_1+float(line[6])/2
			y0 = y_1+float(line[8])/2
			# print(x0,y0)
			# exit()
			jud = float(line[14])
			adv = float(line[16])
			r1 = float(line[10])
			r2 = float(line[12])
			width = float(line[6])
			height = float(line[8])
			s1 = r1*width
			s2 = r2*height

			x1 = x_1
			x2 = x1+width-s1
			x3 = x1+width
			x4 = x1+s1

			y1 = y_1+s2
			y2 = y_1+height
			y3 = y_1+height-s2
			y4 = y_1

			theta = round(math.atan(s2/s1)/2/3.1415926*360,3)

			d1_4 = distance(x1,y1,x4,y4)
			d4_3 = distance(x4,y4,x3,y3)


			pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32) #创建一个坐标数组
			# print(line_x0)
			# exit()
			# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
			#pts = np.array([[x[0],y[0]], [x[1],y[1]], [x[2],y[2]], [x[3],y[3]]], np.int32) #创建一个坐标数组
			pts = pts.reshape((-1, 1, 2)) #矩阵变换
			cv2.polylines(img, [pts], True, (0, 0 , 255),3) #这里注意 pts 要用[]括起来
			
			text = 'adv: '+str(line[16])+'jud'+str(line[14])
			#text1 = 'jud'+str(jud)
			cv2.putText(img, text, (int(x0),int(y0)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)

			ptStart = (int(x0), int(y0))
			#exit()
			thre = 0.1
			adv = 0 if adv < thre else 1
			jud = 0 if jud < thre else 1
			print(adv,jud)
			#exit()

			if r2==1:
				if jud == 1 and adv == 1:
					ptEnd = (int(x2+abs(x3-x2)/2), int(y3+abs(y2-y3)/2))
				if jud == 1 and adv == 0:
					ptEnd = (int(x4+abs(x3-x4)/2), int(y4+abs(y3-y4)/2))
				if jud == 0 and adv == 1:
					ptEnd = (int(x1+abs(x2-x1)/2), int(y1+abs(y2-y1)/2))
				if jud == 0 and adv == 0:
					ptEnd = (int(x1+abs(x4-x1)/2), int(y4+abs(y4-y1)/2))

			else:
				if jud == 1 and adv == 1:
					ptEnd = (int(x4+abs(x3-x4)/2), int(y4+abs(y3-y4)/2))
				if jud == 1 and adv == 0:
					ptEnd = (int(x1+abs(x4-x1)/2), int(y4+abs(y1-y4)/2))
				if jud == 0 and adv == 1:
					ptEnd = (int(x2+abs(x3-x2)/2), int(y3+abs(y2-y3)/2))
				if jud == 0 and adv == 0:
					ptEnd = (int(x1+abs(x2-x1)/2), int(y1+abs(y2-y1)/2))



			# if adv == 1:  #adv = 1  在长轴方向标

			# 	if d1_4>d4_3:

			# 		if jud == 0:
			# 			ptEnd = (int(x1+abs(x2-x1)/2), int(y1+abs(y2-y1)/2))
			# 		else:
			# 			ptEnd = (int(x4+abs(x4-x3)/2), int(y4+abs(y4-y3)/2))

			# 	if d1_4<d4_3:

			# 		if jud == 0:
			# 			ptEnd = (int(x2+abs(x3-x2)/2), int(y3+abs(y3-y2)/2))
			# 		else:
			# 			ptEnd = (int(x1+abs(x4-x1)/2), int(y4+abs(y4-y1)/2))

			# if adv == 0:    #adv = 0   在短轴方向标

			# 	if d1_4>d4_3:

			# 		if jud == 0:
			# 			ptEnd = (int(x2+abs(x3-x2)/2), int(y3+abs(y2-y3)/2))

			# 		else:
						
			# 			ptEnd = (int(x1+abs(x4-x1)/2), int(y4+abs(y1-y4)/2))

			# 	if d1_4<d4_3:

			# 		if jud == 0:
			# 			ptEnd = (int(x1+abs(x2-x1)/2), int(y1+abs(y2-y1)/2))
			# 		else:
			# 			ptEnd = (int(x4+abs(x3-x4)/2), int(y4+abs(y3-y4)/2))



			# if jud < 0.5:
			# 	if d1_4>d4_3:
			# 		ptEnd = (int(x1+abs(x2-x1)/2), int(y1+abs(y2-y1)/2))
			# 	if d1_4<d4_3:
			# 		ptEnd = (int(x2+abs(x3-x2)/2), int(y3+abs(y2-y3)/2))
			#print(ptStart,ptEnd)
			point_color = (0, 255, 0) # BGR
			thickness = 1 
			lineType = 4
			cv2.arrowedLine(img,ptStart, ptEnd, point_color,2,0,0,0.2)
			#cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
			cv2.imwrite(str(out_path[-1]),img)
			cv2.namedWindow('Ryolo')
			cv2.imshow('Ryolo', img)
			cv2.waitKey(1)
			#exit()
		# else:
		# 	continue
			#exit()

# 画多边形

# img = np.zeros((512, 512, 3), np.uint8)
# # def polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None)
# # 参数 pts : 多边形曲线阵列 就是要绘制的多边形的各个点坐标
# # 参数 isClosed : 表示多边形是闭合还是开放的
# pts = np.array([[20,20], [20, 50], [40, 70], [80, 70]], np.int32) #创建一个坐标数组
# # pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1, 1, 2)) #矩阵变换
# cv2.polylines(img, [pts], True, (0, 255 , 0)) #这里注意 pts 要用[]括起来
# cv2.namedWindow('polylines')
# cv2.imshow('polylines', img)
# cv2.waitKey(10000)
