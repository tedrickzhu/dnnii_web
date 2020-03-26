# -*- coding: utf-8 -*-
# @Time    : 20-2-12 上午10:46
# @Author  : zhuzhengyi
import os,time
import cv2

from flask import Blueprint, jsonify, url_for
from flask import render_template,request,redirect
from werkzeug.utils import secure_filename

from App.modelsApi import center_inp,freeform_inp
from App.basedata import BaseData

viewsblue = Blueprint('viewsblue', __name__, template_folder="./templates", static_folder='./static')
# viewsblue = Blueprint('viewsblue', __name__)
basedata = BaseData()

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in basedata.ALLOWED_EXTENSIONS


@viewsblue.route('/')
def index():
	return render_template("index.html")

'''
需要返回页面的数据：
	1，修复结果的路径
	2，修复结果对应的算法以及训练模型所用使用的数据集，例如，gmcnn-pstreetview
	3,输入图片的路径，图片中应有待修复区域的选择
	4，真实图片的路径
'''
@viewsblue.route('/centermask/',methods=["GET","POST"])
def centermask():
	if request.method == "GET":
		return render_template("centermask.html")
	else:
		dataset = request.form.get("dataset")
		testimg = request.form.get("choosedimg")
		imgno = testimg[3:]
		context = center_inp(basedata,dataset,imgno)
		return render_template("centerresult.html",**context)

'''
1，freefrom 页面选择好相关参数后，跳转到画布页面，准备交互
2，将上传图片的尺寸传给canvas，用于初始化canvas的大小
'''
@viewsblue.route('/freeform/',methods=["GET","POST"])
def freeform():
	if request.method == "GET":
		return render_template("freeform.html")
	else:
		#根据name属性获得上传的文件和其他相关参数
		dataset = request.form.get("dataset")
		algrithm = request.form.get("algrithm")
		# f = request.files.get('uploadimgfile')
		f = request.files['uploadimgfile']

		if not (f and allowed_file(f.filename)):
			return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、jpeg、JPEG"})
		# imgname = f.filename.split('.')[0]
		# 注意：没有的文件夹一定要先创建，不然会提示没有该路径
		uploaddir = os.path.join(basedata.UPLOAD_BASE_DIR,'temp')
		if os.path.exists(uploaddir) is False:
			os.mkdir(uploaddir)
		upload_path = os.path.join(uploaddir, secure_filename(f.filename))
		# upload_path = os.path.join(basepath, 'static/images','test.jpg')
		f.save(upload_path)

		# 使用Opencv转换一下图片格式和名称
		img = cv2.imread(upload_path)
		img = cv2.resize(img,(256,256))

		restoredir = os.path.join(basedata.UPLOAD_BASE_DIR,'restore')
		if os.path.exists(restoredir) is False:
			os.mkdir(restoredir)
		restorepath = 'upload/restore/'+time.strftime('%Y%m%d%H%M%S')+'.png'
		cv2.imwrite('./App/static/'+restorepath, img)
		context={
			'imagesize':[int(img.shape[0]),int(img.shape[1])],
			'imagepath':restorepath,
			'dataset':dataset,
			'algrithm':algrithm

		}
		return render_template("freeformcanvas.html",**context)


@viewsblue.route('/freeform/results/',methods=["POST"])
def results():
	if request.method == "POST":
		imagepath=request.form.get("choosedimagepath")
		dataset=request.form.get("chooseddataset")
		algrithm=request.form.get("choosedalgrithm")
		rectmasks=request.form.get("choosedmasks")
		print('this is parameters:',type(rectmasks),rectmasks,imagepath,type(imagepath),type(dataset),dataset,type(algrithm),algrithm)

		#不支持该属性
		# print(request.is_xhr)
		if dataset=='None' or dataset is None or dataset=='':
			dataset='places2'
		if rectmasks is None or rectmasks=='None' or rectmasks=='':
			# context=freeform_inp(basedata,imagepath,dataset,rectmasks,algrithm)
			context = {
				'imagesize': [256,256],
				'imagepath': imagepath,
				'dataset': dataset,
				'algrithm': algrithm
			}
			return render_template("freeformcanvas.html", **context)

		context=freeform_inp(basedata,imagepath,dataset,rectmasks,algrithm)

		return render_template("freeformresult.html",**context)
		# return redirect(url_for("viewsblue.results"))