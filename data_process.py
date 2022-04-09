{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f25ca731",
   "metadata": {
    "toc": true
   },
   "source": [
    "<h1>Table of Contents<span class=\"tocSkip\"></span></h1>\n",
    "<div class=\"toc\"><ul class=\"toc-item\"></ul></div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c28ff2a8",
   "metadata": {},
   "source": [
    "# 图像合并"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cff44d3a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2 as cv\n",
    "import glob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "139aa1f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_path = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.05_M_U1U2HIGHout\\*.jpg\")\n",
    "img_path.sort(key=lambda x:int(x.split('\\\\')[7].split('_')[1]))\n",
    "img_low_high = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.05_M_U1_low\\*.jpg\")\n",
    "img_low_high.sort(key=lambda x:int(x.split('\\\\')[7].split('_')[1].split(\".\")[0]))\n",
    "save_path = r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.05_M_U1U2HIGHout_low/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de4188aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_path[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95c94952",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_low_high[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ccee92c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def tw0_one(path,path_low):\n",
    "    img = cv.imread(path)\n",
    "    low = cv.imread(path_low)\n",
    "    fu  = img+0.3*low\n",
    "    cv.imwrite(save_path+str(path.split(\"\\\\\")[7].split(\"s\")[0])+\".jpg\",fu)\n",
    "for path,path_low in zip(img_path,img_low_high):\n",
    "    tw0_one(path,path_low)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ef8cbaa",
   "metadata": {},
   "source": [
    "# 图片拼接"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f838179c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2 as cv\n",
    "import glob\n",
    "img_path = glob.glob(r\"F:\\Videodenoise\\dataset\\train\\GS\\GS_NOISE_F\\GS30_F\\GSnoise30_F_U1U2HIGHout_low_out\\*.jpg\")\n",
    "img_path.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[1]))\n",
    "save_path = r\"F:\\Videodenoise\\dataset\\train\\GS\\GS_NOISE_F\\GS30_F\\GSnoise30_F_U1U2HIGHout_low_outimages/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2a015ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_path[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "163e615a",
   "metadata": {},
   "outputs": [],
   "source": [
    "ip = []\n",
    "for i in img_path:\n",
    "    ip.append(int(i.split(\"\\\\\")[8].split(\"_\")[0]))\n",
    "    ip = list(set(ip))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "516708f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "all_ip = []\n",
    "ip_=[]\n",
    "for m in range(len(ip)):\n",
    "    one_img_id = []\n",
    "    for i in img_path:\n",
    "        if int(i.split(\"\\\\\")[8].split(\"_\")[0]) == int(ip[m]):\n",
    "            ip_.append(int(ip[m]))\n",
    "            one_img_id.append(i)\n",
    "        else:\n",
    "            pass\n",
    "    all_ip.append(one_img_id)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3bf1561",
   "metadata": {},
   "outputs": [],
   "source": [
    "ip_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "128946bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "for j in range(len(img_path)):  \n",
    "    i=0\n",
    "    h1 = cv.hconcat([cv.imread(all_ip[j][i],1),cv.imread(all_ip[j][i+1],1),cv.imread(all_ip[j][i+2],1),cv.imread(all_ip[j][i+3],1)])\n",
    "    h2 = cv.hconcat([cv.imread(all_ip[j][i+4],1),cv.imread(all_ip[j][i+5],1),cv.imread(all_ip[j][i+6],1),cv.imread(all_ip[j][i+7],1)]) \n",
    "    h3 = cv.hconcat([cv.imread(all_ip[j][i+8],1),cv.imread(all_ip[j][i+9],1),cv.imread(all_ip[j][i+10],1),cv.imread(all_ip[j][i+11],1)])\n",
    "    h2_ = cv.vconcat([h1,h2])\n",
    "    h4 = cv.vconcat([h2_,h3])\n",
    "    cv.imwrite(save_path+str(all_ip[j][0].split(\"\\\\\")[8].split(\"_\")[0])+\".jpg\",h4)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e5d1d48",
   "metadata": {},
   "source": [
    "# 图片RESIZE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1888f040",
   "metadata": {},
   "outputs": [],
   "source": [
    "import glob\n",
    "import cv2 as cv"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "408bfbac",
   "metadata": {},
   "source": [
    "img_path1 = glob.glob(r\"F:\\Videodenoise\\dataset\\train\\org\\*.jpg\")\n",
    "save_path1 = r\"F:\\Videodenoise\\dataset\\train\\org512/\"\n",
    "'''save_path2 = r\"D:\\Research Related\\Paper Related\\Step-by-step image generation\\Code\\dataset\\images256/\"\n",
    "img_path2 = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise25_F\\GSnoise25_F_U1U2HIGHout_low_outIMAGES\\*.jpg\")\n",
    "save_path2 = r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise25_F\\GSnoise25_F_U1U2HIGHout_low_outIMAGES1024_512/\"\n",
    "\n",
    "img_path3 = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise30_F\\GSnoise30_F_U1U2HIGHout_low_outIMAGES\\*.jpg\")\n",
    "save_path3 = r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise30_F\\GSnoise30_F_U1U2HIGHout_low_outIMAGES1024_512/\"\n",
    "\n",
    "img_path4 = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise50_F\\GSnoise50_F_U1U2HIGHout_low_outIMAGES\\*.jpg\")\n",
    "save_path4 = r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise50_F\\GSnoise50_F_U1U2HIGHout_low_outIMAGES1024_512/\"\n",
    "\n",
    "img_path5 = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.1_M_U1U2HIGHout_low_outIMAGES\\*.jpg\")\n",
    "save_path5 = r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.1_M_U1U2HIGHout_low_outIMAGES512_1024/\"\n",
    "\n",
    "img_path6 = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.05_M_U1U2HIGHout_low_outIMAGES\\*.jpg\")\n",
    "save_path6 = r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.05_M_U1U2HIGHout_low_outIMAGES512_1024/\"\n",
    "\n",
    "img_path7 = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\test_SET_ORG640_480\\*.jpg\")\n",
    "save_path7 = r\"F:\\Videodenoise\\dataset\\test\\test_SET_ORG512_1024/\"'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8dee5f32",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_path1[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "afcce8ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "def Resize(img_path,save_path,shape1,shape2,m):\n",
    "    img = cv.imread(img_path,1)\n",
    "    cv.imwrite(save_path+str(img_path.split(\"\\\\\")[m].split(\".\")[0])+\".jpg\",cv.resize(img,(shape1,shape2)))"
   ]
  },
  {
   "cell_type": "raw",
   "id": "1c513e05",
   "metadata": {},
   "source": [
    "for i in range(len(img_path1)):\n",
    "    Resize(img_path1[i],save_path1,1024,512,8)\n",
    "for i in range(len(img_path2)):\n",
    "    Resize(img_path2[i],save_path2,1024,512,8)\n",
    "for i in range(len(img_path3)):\n",
    "    Resize(img_path3[i],save_path3,1024,512,8)\n",
    "for i in range(len(img_path4)):\n",
    "    Resize(img_path4[i],save_path4,1024,512,8)\n",
    "for i in range(len(img_path5)):\n",
    "    Resize(img_path5[i],save_path5,1024,512,7)\n",
    "for i in range(len(img_path6)):\n",
    "    Resize(img_path6[i],save_path6,1024,512,7)\n",
    "for i in range(len(img_path7)):\n",
    "    Resize(img_path7[i],save_path7,1024,512,5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f660f2b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_path1 = glob.glob(r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise50_F\\GSnoise50_F_U1U2HIGHout_low_outIMAGESU3\\*.jpg\")\n",
    "save_path1 = r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise50_F\\GSnoise50_F_U1U2HIGHout_low_outIMAGESU3480/\"\n",
    "for i in range(len(img_path1)):\n",
    "    Resize(img_path1[i],save_path1,640,480,8)\n",
    "    #Resize(img_path1[i],save_path2,256,256,7)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9058a4e1",
   "metadata": {},
   "source": [
    "# PSNR和SSIM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bbee870",
   "metadata": {},
   "outputs": [],
   "source": [
    "import glob \n",
    "import tensorflow as tf\n",
    "import math\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ba198ae",
   "metadata": {
    "cell_style": "center"
   },
   "outputs": [],
   "source": [
    "GS12_15 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_15\\*.jpg\")\n",
    "GS12_15.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_15F = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_15F\\*.jpg\")\n",
    "GS12_15F.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_15FU1 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_15FU1\\*.jpg\")\n",
    "GS12_15FU1.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_15FU1PATCH = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_15FU1PATCH\\*.jpg\")\n",
    "GS12_15FU1PATCH.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[1].split('.')[0]))\n",
    "GS12_15FU1U2U3 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_15FU1U2U3\\*.jpg\")\n",
    "GS12_15FU1U2U3.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[1].split('.')[0]))\n",
    "GS12_15FU1U2U3IMAGE = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_15FU1U2U3IMAGE\\*.jpg\")\n",
    "GS12_15FU1U2U3IMAGE.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_15FU1U2U3_OP = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_15FU1U2U3_OP\\*.jpg\")\n",
    "GS12_15FU1U2U3_OP.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "\n",
    "GS12_25 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_25\\*.jpg\")\n",
    "GS12_25.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_25F = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_25F\\*.jpg\")\n",
    "GS12_25F.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_25FU1 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_25FU1\\*.jpg\")\n",
    "GS12_25FU1.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_25FU1PATCH = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_25FU1PATCH\\*.jpg\")\n",
    "GS12_25FU1PATCH.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[1].split('.')[0]))\n",
    "GS12_25FU1U2U3 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_25FU1U2U3\\*.jpg\")\n",
    "GS12_25FU1U2U3.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[1].split('.')[0]))\n",
    "GS12_25FU1U2U3IMAGE = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_25FU1U2U3IMAGE\\*.jpg\")\n",
    "GS12_25FU1U2U3IMAGE.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_25FU1U2U3_OP = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_25FU1U2U3_OP\\*.jpg\")\n",
    "GS12_25FU1U2U3_OP.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "\n",
    "GS12_30 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_30\\*.jpg\")\n",
    "GS12_30.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_30F = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_30F\\*.jpg\")\n",
    "GS12_30F.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_30FU1 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_30FU1\\*.jpg\")\n",
    "GS12_30FU1.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_30FU1PATCH = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_30FU1PATCH\\*.jpg\")\n",
    "GS12_30FU1PATCH.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "GS12_30FU1U2U3 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_30FU1U2U3\\*.jpg\")\n",
    "GS12_30FU1U2U3.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "GS12_30FU1U2U3IMAGE = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_30FU1U2U3IMAGE\\*.jpg\")\n",
    "GS12_30FU1U2U3IMAGE.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_30FU1U2U3_OP = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_30FU1U2U3_OP\\*.jpg\")\n",
    "GS12_30FU1U2U3_OP.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "\n",
    "GS12_50 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_50\\*.jpg\")\n",
    "GS12_50.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_50F = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_50F\\*.jpg\")\n",
    "GS12_50F.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_50FU1 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_50FU1\\*.jpg\")\n",
    "GS12_50FU1.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_50FU1PATCH = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_50FU1PATCH\\*.jpg\")\n",
    "GS12_50FU1PATCH.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "GS12_50FU1U2U3 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_50FU1U2U3\\*.jpg\")\n",
    "GS12_50FU1U2U3.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "GS12_50FU1U2U3IMAGE = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_50FU1U2U3IMAGE\\*.jpg\")\n",
    "GS12_50FU1U2U3IMAGE.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "GS12_50FU1U2U3_OP = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\GS12_50FU1U2U3_OP\\*.jpg\")\n",
    "GS12_50FU1U2U3_OP.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "\n",
    "JY12_01 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.1\\*.jpg\")\n",
    "JY12_01.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "JY12_01F = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.1F\\*.jpg\")\n",
    "JY12_01F.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "JY12_01FU1 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.1FU1\\*.jpg\")\n",
    "JY12_01FU1.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "JY12_01FU1PATCH = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.1FU1PATCH\\*.jpg\")\n",
    "JY12_01FU1PATCH.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "JY12_01FU1U2U3 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.1FU1U2U3\\*.jpg\")\n",
    "JY12_01FU1U2U3.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "JY12_01FU1U2U3IMAGE = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.1FU1U2U3IMAGES\\*.jpg\")\n",
    "JY12_01FU1U2U3IMAGE.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "JY12_01FU1U2U3_OP = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.1FU1U2U3_OP\\*.jpg\")\n",
    "JY12_01FU1U2U3_OP.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "\n",
    "JY12_005 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.05\\*.jpg\")\n",
    "JY12_005.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "JY12_005F = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.05F\\*.jpg\")\n",
    "JY12_005F.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "JY12_005FU1 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.05FU1\\*.jpg\")\n",
    "JY12_005FU1.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "JY12_005FU1PATCH = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.05FU1PATCH\\*.jpg\")\n",
    "JY12_005FU1PATCH.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "JY12_005FU1U2U3 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.05FU1U2U3\\*.jpg\")\n",
    "JY12_005FU1U2U3.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "JY12_005FU1U2U3IMAGE = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.05FU1U2U3IMAGES\\*.jpg\")\n",
    "JY12_005FU1U2U3IMAGE.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "JY12_005FU1U2U3_OP = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\JY12_0.05FU1U2U3_OP\\*.jpg\")\n",
    "JY12_005FU1U2U3_OP.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[0]))\n",
    "\n",
    "org12_P = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\org12_P\\*.jpg\")\n",
    "org12_P.sort(key=lambda x:int(x.split('\\\\')[8].split('_')[1].split('.')[0]))\n",
    "\n",
    "\n",
    "org12 = glob.glob(r\"C:\\Users\\Administrator\\Desktop\\Videodenoise\\文中图\\test_Ablation experiments\\org12\\*.jpg\")\n",
    "org12.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16158fbb",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(GS12_15),len(GS12_25),len(GS12_30),len(GS12_50),len(JY12_01),len(JY12_005)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8cd1d6c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(GS12_15F),len(GS12_25F),len(GS12_30F),len(GS12_50F),len(JY12_01F),len(JY12_005F)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a62359e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(GS12_15FU1),len(GS12_25FU1),len(GS12_30FU1),len(GS12_50FU1),len(JY12_01FU1),len(JY12_005FU1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b60e6f11",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(GS12_15FU1PATCH),len(GS12_25FU1PATCH),len(GS12_30FU1PATCH),len(GS12_50FU1PATCH),len(JY12_01FU1PATCH),len(JY12_005FU1PATCH)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8ebbba1",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(GS12_15FU1U2U3),len(GS12_15FU1U2U3),len(GS12_15FU1U2U3),len(GS12_15FU1U2U3),len(JY12_01FU1U2U3),len(JY12_005FU1U2U3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0bb28cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(GS12_15FU1U2U3IMAGE),len(GS12_15FU1U2U3IMAGE),len(GS12_15FU1U2U3IMAGE),len(GS12_15FU1U2U3IMAGE),len(JY12_01FU1U2U3IMAGE),len(JY12_005FU1U2U3IMAGE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3cc640c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(GS12_15FU1U2U3_OP),len(GS12_15FU1U2U3_OP),len(GS12_15FU1U2U3_OP),len(GS12_15FU1U2U3_OP),len(JY12_01FU1U2U3_OP),len(JY12_005FU1U2U3_OP)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "633b0585",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(org12_P),len(org12)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e1e90a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "    GS12_15P = []\n",
    "    GS12_15FP = []\n",
    "    GS12_15FU1P = []\n",
    "    GS12_15FU1PATCHP = []\n",
    "    GS12_15FU1U2U3P = []\n",
    "    GS12_15FU1U2U3_OPP = []\n",
    "    \n",
    "    GS12_15S = []\n",
    "    GS12_15FS = []\n",
    "    GS12_15FU1S = []\n",
    "    GS12_15FU1PATCHS = []\n",
    "    GS12_15FU1U2U3S = []\n",
    "    GS12_15FU1U2U3_OPS = []\n",
    "    \n",
    "    \n",
    "    \n",
    "    GS12_25P = []\n",
    "    GS12_25FP = []\n",
    "    GS12_25FU1P = []\n",
    "    GS12_25FU1PATCHP = []\n",
    "    GS12_25FU1U2U3P = []\n",
    "    GS12_25FU1U2U3_OPP = []\n",
    "    \n",
    "    GS12_25S = []\n",
    "    GS12_25FS = []\n",
    "    GS12_25FU1S = []\n",
    "    GS12_25FU1PATCHS = []\n",
    "    GS12_25FU1U2U3S = []\n",
    "    GS12_25FU1U2U3_OPS = []\n",
    "    \n",
    "    \n",
    "    GS12_30P = []\n",
    "    GS12_30FP = []\n",
    "    GS12_30FU1P = []\n",
    "    GS12_30FU1PATCHP = []\n",
    "    GS12_30FU1U2U3P = []\n",
    "    GS12_30FU1U2U3_OPP = []\n",
    "    \n",
    "    GS12_30S = []\n",
    "    GS12_30FS = []\n",
    "    GS12_30FU1S = []\n",
    "    GS12_30FU1PATCHS = []\n",
    "    GS12_30FU1U2U3S = []\n",
    "    GS12_30FU1U2U3_OPS = []\n",
    "    \n",
    "    \n",
    "    GS12_50P = []\n",
    "    GS12_50FP = []\n",
    "    GS12_50FU1P = []\n",
    "    GS12_50FU1PATCHP = []\n",
    "    GS12_50FU1U2U3P = []\n",
    "    GS12_50FU1U2U3_OPP = []\n",
    "    \n",
    "    GS12_50S = []\n",
    "    GS12_50FS = []\n",
    "    GS12_50FU1S = []\n",
    "    GS12_50FU1PATCHS = []\n",
    "    GS12_50FU1U2U3S = []\n",
    "    GS12_50FU1U2U3_OPS = []\n",
    "    \n",
    "    jy0112_1P = []\n",
    "    jy0112_1FP = []\n",
    "    jy0112_1FU1P = []\n",
    "    jy0112_1FU1PATCHP = []\n",
    "    jy0112_1FU1U2U3P = []\n",
    "    jy0112_1FU1U2U3_OPP = []\n",
    "    \n",
    "    jy0112_1S = []\n",
    "    jy0112_1FS = []\n",
    "    jy0112_1FU1S = []\n",
    "    jy0112_1FU1PATCHS = []\n",
    "    jy0112_1FU1U2U3S = []\n",
    "    jy0112_1FU1U2U3_OPS = []\n",
    "    \n",
    "    jy005P = []\n",
    "    jy005FP = []\n",
    "    jy005FU1P = []\n",
    "    jy005FU1PATCHP = []\n",
    "    jy005FU1U2U3P = []\n",
    "    jy005FU1U2U3_OPP = []\n",
    "    \n",
    "    jy005S = []\n",
    "    jy005FS = []\n",
    "    jy005FU1S = []\n",
    "    jy005FU1PATCHS = []\n",
    "    jy005FU1U2U3S = []\n",
    "    jy005FU1U2U3_OPS = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fabc63c2",
   "metadata": {
    "cell_style": "center"
   },
   "outputs": [],
   "source": [
    "def load_and_preprocess1(path1,path2):   \n",
    "    img1= tf.io.read_file(path1)\n",
    "    img1 = tf.image.decode_jpeg(img1,channels=3)\n",
    "    img1 = tf.image.resize(img1,(sahpe1,sahpe2))\n",
    "    img2 = tf.io.read_file(path2)\n",
    "    img2 = tf.image.decode_jpeg(img2,channels=3) \n",
    "    img2 = tf.image.resize(img2,(sahpe1,sahpe2))\n",
    "    psnr = tf.image.psnr(img1,img2,max_val=255).numpy()\n",
    "    ssim = tf.image.ssim(img1,img2,max_val=1).numpy()\n",
    "    return psnr,ssim\n",
    "\n",
    "\n",
    "def com_df():\n",
    "    for i in range(len(org12_P)):\n",
    "        GS12_15p,GS12_15s = load_and_preprocess1(GS12_15FU1PATCH[i],org12_P[i])\n",
    "        GS12_15Fp,GS12_15Fs = load_and_preprocess1(GS12_15FU1U2U3[i],org12_P[i])\n",
    "        \n",
    "        GS12_15FU1PATCHP.append(GS12_15p)\n",
    "        GS12_15FU1U2U3P.append(GS12_15Fp)\n",
    "       \n",
    "        GS12_15FU1PATCHS.append(GS12_15s)\n",
    "        GS12_15FU1U2U3S.append(GS12_15Fs)\n",
    "        \n",
    "        \n",
    "        \n",
    "        \n",
    "        GS12_25p,GS12_25s = load_and_preprocess1(GS12_25FU1PATCH[i],org12_P[i])\n",
    "        GS12_25Fp,GS12_25Fs = load_and_preprocess1(GS12_25FU1U2U3[i],org12_P[i])\n",
    "        \n",
    "        GS12_25FU1PATCHP.append(GS12_25p)\n",
    "        GS12_25FU1U2U3P.append(GS12_25Fp)\n",
    "       \n",
    "        GS12_25FU1PATCHS.append(GS12_25s)\n",
    "        GS12_25FU1U2U3S.append(GS12_25Fs)\n",
    "        \n",
    "        \n",
    "        \n",
    "        \n",
    "        GS12_30p,GS12_30s = load_and_preprocess1(GS12_30FU1PATCH[i],org12_P[i])\n",
    "        GS12_30Fp,GS12_30Fs = load_and_preprocess1(GS12_30FU1U2U3[i],org12_P[i])\n",
    "        \n",
    "        GS12_30FU1PATCHP.append(GS12_30p)\n",
    "        GS12_30FU1U2U3P.append(GS12_30Fp)\n",
    "       \n",
    "        GS12_30FU1PATCHS.append(GS12_30s)\n",
    "        GS12_30FU1U2U3S.append(GS12_30Fs)\n",
    "        \n",
    "        \n",
    "        \n",
    "\n",
    "        GS12_50p,GS12_50s = load_and_preprocess1(GS12_50FU1PATCH[i],org12_P[i])\n",
    "        GS12_50Fp,GS12_50Fs = load_and_preprocess1(GS12_50FU1U2U3[i],org12_P[i])\n",
    "        \n",
    "        GS12_50FU1PATCHP.append(GS12_50p)\n",
    "        GS12_50FU1U2U3P.append(GS12_50Fp)\n",
    "       \n",
    "        GS12_50FU1PATCHS.append(GS12_50s)\n",
    "        GS12_50FU1U2U3S.append(GS12_50Fs)\n",
    "        \n",
    "\n",
    "        \n",
    "        \n",
    "        JY12_01p,JY12_01s = load_and_preprocess1(JY12_01FU1PATCH[i],org12_P[i])\n",
    "        JY12_01Fp,JY12_01Fs = load_and_preprocess1(JY12_01FU1U2U3[i],org12_P[i])\n",
    "        \n",
    "        jy0112_1FU1PATCHP.append(JY12_01p)\n",
    "        jy0112_1FU1U2U3P.append(JY12_01Fp)\n",
    "       \n",
    "        jy0112_1FU1PATCHS.append(JY12_01s )\n",
    "        jy0112_1FU1U2U3S.append(JY12_01Fs)\n",
    "        \n",
    "        \n",
    "        \n",
    "        JY005p,JY005s = load_and_preprocess1(JY12_005FU1PATCH[i],org12_P[i])\n",
    "        JY005Fp,JY005Fs = load_and_preprocess1(JY12_005FU1U2U3[i],org12_P[i])\n",
    "        \n",
    "        jy005FU1PATCHP.append(JY005p)\n",
    "        jy005FU1U2U3P.append(JY005Fp)\n",
    "       \n",
    "        jy005FU1PATCHS.append(JY005s )\n",
    "        jy005FU1U2U3S.append(JY005Fs)\n",
    "            \n",
    "    for i in range(len(org12)):\n",
    "        GS12_15p,GS12_15s = load_and_preprocess1(GS12_15[i],org12[i])\n",
    "        GS12_15Fp,GS12_15Fs = load_and_preprocess1(GS12_15F[i],org12[i])\n",
    "        GS12_15FU1p,GS12_15FU1s = load_and_preprocess1(GS12_15FU1[i],org12[i])\n",
    "        GS12_15FU1U2U3_OPp,GS12_15FU1U2U3_OPs = load_and_preprocess1( GS12_15FU1U2U3IMAGE[i],GS12_15FU1U2U3_OP[i])\n",
    "        \n",
    "        GS12_15P.append(GS12_15p)\n",
    "        GS12_15FP.append(GS12_15Fp)\n",
    "        GS12_15FU1P.append(GS12_15FU1p)\n",
    "        GS12_15FU1U2U3_OPP.append(GS12_15FU1U2U3_OPp)\n",
    "        \n",
    "        GS12_15S.append(GS12_15s)\n",
    "        GS12_15FS.append(GS12_15Fs)\n",
    "        GS12_15FU1S.append(GS12_15FU1s)\n",
    "        GS12_15FU1U2U3_OPS.append(GS12_15FU1U2U3_OPs)\n",
    "        \n",
    "        \n",
    "        \n",
    "        GS12_25p,GS12_25s = load_and_preprocess1(GS12_25[i],org12[i])\n",
    "        GS12_25Fp,GS12_25Fs = load_and_preprocess1(GS12_25F[i],org12[i])\n",
    "        GS12_25FU1p,GS12_25FU1s = load_and_preprocess1(GS12_25FU1[i],org12[i])\n",
    "        GS12_25FU1U2U3_OPp,GS12_25FU1U2U3_OPs = load_and_preprocess1( GS12_25FU1U2U3IMAGE[i],GS12_25FU1U2U3_OP[i])\n",
    "        \n",
    "        GS12_25P.append(GS12_25p)\n",
    "        GS12_25FP.append(GS12_25Fp)\n",
    "        GS12_25FU1P.append(GS12_25FU1p)\n",
    "        GS12_25FU1U2U3_OPP.append(GS12_25FU1U2U3_OPp)\n",
    "        \n",
    "        GS12_25S.append(GS12_25s)\n",
    "        GS12_25FS.append(GS12_25Fs)\n",
    "        GS12_25FU1S.append(GS12_25FU1s)\n",
    "        GS12_25FU1U2U3_OPS.append(GS12_25FU1U2U3_OPs)\n",
    " \n",
    "\n",
    "        GS12_30p,GS12_30s = load_and_preprocess1(GS12_30[i],org12[i])\n",
    "        GS12_30Fp,GS12_30Fs = load_and_preprocess1(GS12_30F[i],org12[i])\n",
    "        GS12_30FU1p,GS12_30FU1s = load_and_preprocess1(GS12_30FU1[i],org12[i])\n",
    "        GS12_30FU1U2U3_OPp,GS12_30FU1U2U3_OPs = load_and_preprocess1( GS12_30FU1U2U3IMAGE[i],GS12_30FU1U2U3_OP[i])\n",
    "        \n",
    "        GS12_30P.append(GS12_30p)\n",
    "        GS12_30FP.append(GS12_30Fp)\n",
    "        GS12_30FU1P.append(GS12_30FU1p)\n",
    "        GS12_30FU1U2U3_OPP.append(GS12_30FU1U2U3_OPp)\n",
    "        \n",
    "        GS12_30S.append(GS12_30s)\n",
    "        GS12_30FS.append(GS12_30Fs)\n",
    "        GS12_30FU1S.append(GS12_30FU1s)\n",
    "        GS12_30FU1U2U3_OPS.append(GS12_30FU1U2U3_OPs)\n",
    "\n",
    "       \n",
    "        \n",
    "        GS12_50p,GS12_50s = load_and_preprocess1(GS12_50[i],org12[i])\n",
    "        GS12_50Fp,GS12_50Fs = load_and_preprocess1(GS12_50F[i],org12[i])\n",
    "        GS12_50FU1p,GS12_50FU1s = load_and_preprocess1(GS12_50FU1[i],org12[i])\n",
    "        GS12_50FU1U2U3_OPp,GS12_50FU1U2U3_OPs = load_and_preprocess1( GS12_50FU1U2U3IMAGE[i],GS12_50FU1U2U3_OP[i])\n",
    "        \n",
    "        GS12_50P.append(GS12_50p)\n",
    "        GS12_50FP.append(GS12_50Fp)\n",
    "        GS12_50FU1P.append(GS12_50FU1p)\n",
    "        GS12_50FU1U2U3_OPP.append(GS12_50FU1U2U3_OPp)\n",
    "        \n",
    "        GS12_50S.append(GS12_50s)\n",
    "        GS12_50FS.append(GS12_50Fs)\n",
    "        GS12_50FU1S.append(GS12_50FU1s)\n",
    "        GS12_50FU1U2U3_OPS.append(GS12_50FU1U2U3_OPs)\n",
    " \n",
    "        \n",
    "        JY12_01p,JY12_01s = load_and_preprocess1(JY12_01[i],org12[i])\n",
    "        JY12_01Fp,JY12_01Fs = load_and_preprocess1(JY12_01F[i],org12[i])\n",
    "        JY12_01FU1p,JY12_01FU1s = load_and_preprocess1(JY12_01FU1[i],org12[i])\n",
    "        JY12_01FU1U2U3_OPp,JY12_01FU1U2U3_OPs = load_and_preprocess1( JY12_01FU1U2U3IMAGE[i],JY12_01FU1U2U3_OP[i])\n",
    "        \n",
    "        jy0112_1P.append(JY12_01p)\n",
    "        jy0112_1FP.append(JY12_01Fp)\n",
    "        jy0112_1FU1P.append(JY12_01FU1p)\n",
    "        jy0112_1FU1U2U3_OPP.append(JY12_01FU1U2U3_OPp)\n",
    "        \n",
    "        jy0112_1S.append(JY12_01s)\n",
    "        jy0112_1FS.append(JY12_01Fs)\n",
    "        jy0112_1FU1S.append(JY12_01FU1s)\n",
    "        jy0112_1FU1U2U3_OPS.append(JY12_01FU1U2U3_OPs)\n",
    "\n",
    "        \n",
    "\n",
    "        JY12_005p,JY12_005s = load_and_preprocess1(JY12_005[i],org12[i])\n",
    "        JY12_005Fp,JY12_005Fs = load_and_preprocess1(JY12_005F[i],org12[i])\n",
    "        JY12_005FU1p,JY12_005FU1s = load_and_preprocess1(JY12_005FU1[i],org12[i])\n",
    "        JY12_005FU1U2U3_OPp,JY12_005FU1U2U3_OPs = load_and_preprocess1( JY12_005FU1U2U3IMAGE[i],JY12_005FU1U2U3_OP[i])\n",
    "        \n",
    "        jy005P.append(JY12_005p)\n",
    "        jy005FP.append(JY12_005Fp)\n",
    "        jy005FU1P.append(JY12_005FU1p)\n",
    "        jy005FU1U2U3_OPP.append(JY12_005FU1U2U3_OPp)\n",
    "        \n",
    "        jy005S.append(JY12_005s)\n",
    "        jy005FS.append(JY12_005Fs)\n",
    "        jy005FU1S.append(JY12_005FU1s)\n",
    "        jy005FU1U2U3_OPS.append(JY12_005FU1U2U3_OPs)\n",
    "\n",
    "        \n",
    "    dic_psnr_images={\n",
    "        \"GS12_15P\":GS12_15P,\n",
    "        \"GS12_15FP\":GS12_15FP,\n",
    "        \"GS12_15FU1P\":GS12_15FU1P,\n",
    "        #\"GS12_15FU1PATCHP\":GS12_15FU1PATCHP,\n",
    "        #\"GS12_15FU1U2U3P\":GS12_15FU1U2U3P,\n",
    "        \"GS12_15FU1U2U3_OPP\":GS12_15FU1U2U3_OPP,\n",
    "        \n",
    "        \"GS12_25P\":GS12_25P,\n",
    "        \"GS12_25FP\":GS12_25FP,\n",
    "        \"GS12_25FU1P\":GS12_25FU1P,\n",
    "        #\"GS12_25FU1PATCHP\":GS12_25FU1PATCHP,\n",
    "        #\"GS12_25FU1U2U3P\":GS12_25FU1U2U3P,\n",
    "        \"GS12_25FU1U2U3_OPP\":GS12_25FU1U2U3_OPP,\n",
    "        \n",
    "        \n",
    "        \"GS12_30P\":GS12_30P,\n",
    "        \"GS12_30FP\":GS12_30FP,\n",
    "        \"GS12_30FU1P\":GS12_30FU1P,\n",
    "        #\"GS12_30FU1PATCHP\":GS12_30FU1PATCHP,\n",
    "        #\"GS12_30FU1U2U3P\":GS12_30FU1U2U3P,\n",
    "        \"GS12_30FU1U2U3_OPP\":GS12_30FU1U2U3_OPP,\n",
    "        \n",
    "        \n",
    "        \"GS12_50P\":GS12_50P,\n",
    "        \"GS12_50FP\":GS12_50FP,\n",
    "        \"GS12_50FU1P\":GS12_50FU1P,\n",
    "        #\"GS12_50FU1PATCHP\":GS12_50FU1PATCHP,\n",
    "        #\"GS12_50FU1U2U3P\":GS12_50FU1U2U3P,\n",
    "        \"GS12_50FU1U2U3_OPP\":GS12_50FU1U2U3_OPP,\n",
    "\n",
    "\n",
    "        \"jy0112_1P\":jy0112_1P,\n",
    "        \"jy0112_1FP\":jy0112_1FP,\n",
    "        \"jy0112_1FU1P\":jy0112_1FU1P,\n",
    "        #\"jy0112_1FU1PATCHP\":jy0112_1FU1PATCHP,\n",
    "        #\"jy0112_1FU1U2U3P\":jy0112_1FU1U2U3P,\n",
    "        \"jy0112_1FU1U2U3_OPP\":jy0112_1FU1U2U3_OPP,\n",
    "        \n",
    "        \"jy005P\":jy005P,\n",
    "        \"jy005FP\":jy005FP,\n",
    "        \"jy005FU1P\":jy005FU1P,\n",
    "        #\"jy005FU1PATCHP\":jy005FU1PATCHP,\n",
    "        #\"jy005FU1U2U3P\":jy005FU1U2U3P,\n",
    "        \"jy005FU1U2U3_OPP\":jy005FU1U2U3_OPP,\n",
    "        \n",
    "     \n",
    "    }\n",
    "    \n",
    "    dic_psnr_patchs={\n",
    "        #\"GS12_15P\":GS12_15P,\n",
    "        #\"GS12_15FP\":GS12_15FP,\n",
    "        #\"GS12_15FU1P\":GS12_15FU1P,\n",
    "        \"GS12_15FU1PATCHP\":GS12_15FU1PATCHP,\n",
    "        \"GS12_15FU1U2U3P\":GS12_15FU1U2U3P,\n",
    "        #\"GS12_15FU1U2U3_OPP\":GS12_15FU1U2U3_OPP,\n",
    "        \n",
    "        #\"GS12_25P\":GS12_25P,\n",
    "        #\"GS12_25FP\":GS12_25FP,\n",
    "        #\"GS12_25FU1P\":GS12_25FU1P,\n",
    "        \"GS12_25FU1PATCHP\":GS12_25FU1PATCHP,\n",
    "        \"GS12_25FU1U2U3P\":GS12_25FU1U2U3P,\n",
    "        #\"GS12_25FU1U2U3_OPP\":GS12_25FU1U2U3_OPP,\n",
    "        \n",
    "        \n",
    "        #\"GS12_30P\":GS12_30P,\n",
    "        #\"GS12_30FP\":GS12_30FP,\n",
    "        #\"GS12_30FU1P\":GS12_30FU1P,\n",
    "        \"GS12_30FU1PATCHP\":GS12_30FU1PATCHP,\n",
    "        \"GS12_30FU1U2U3P\":GS12_30FU1U2U3P,\n",
    "        #\"GS12_30FU1U2U3_OPP\":GS12_30FU1U2U3_OPP,\n",
    "        \n",
    "        \n",
    "        #\"GS12_50P\":GS12_50P,\n",
    "        #\"GS12_50FP\":GS12_50FP,\n",
    "        #\"GS12_50FU1P\":GS12_50FU1P,\n",
    "        \"GS12_50FU1PATCHP\":GS12_50FU1PATCHP,\n",
    "        \"GS12_50FU1U2U3P\":GS12_50FU1U2U3P,\n",
    "        #\"GS12_50FU1U2U3_OPP\":GS12_50FU1U2U3_OPP,\n",
    "\n",
    "\n",
    "        #\"jy0112_1P\":jy0112_1P,\n",
    "        #\"jy0112_1FP\":jy0112_1FP,\n",
    "        #\"jy0112_1FU1P\":jy0112_1FU1P,\n",
    "        \"jy0112_1FU1PATCHP\":jy0112_1FU1PATCHP,\n",
    "        \"jy0112_1FU1U2U3P\":jy0112_1FU1U2U3P,\n",
    "        #\"jy0112_1FU1U2U3_OPP\":jy0112_1FU1U2U3_OPP,\n",
    "        \n",
    "        #\"jy005P\":jy005P,\n",
    "        #\"jy005FP\":jy005FP,\n",
    "        #\"jy005FU1P\":jy005FU1P,\n",
    "        \"jy005FU1PATCHP\":jy005FU1PATCHP,\n",
    "        \"jy005FU1U2U3P\":jy005FU1U2U3P,\n",
    "        #\"jy005FU1U2U3_OPP\":jy005FU1U2U3_OPP,\n",
    "        \n",
    "     \n",
    "    }\n",
    "    \n",
    "    dic_ssim_images={\n",
    "        \"GS12_15S\":GS12_15S,\n",
    "        \"GS12_15FS\":GS12_15FS,\n",
    "        \"GS12_15FU1S\":GS12_15FU1S,\n",
    "        #\"GS12_15FU1PATCHS\":GS12_15FU1PATCHS,\n",
    "        #\"GS12_15FU1U2U3S\":GS12_15FU1U2U3S,\n",
    "        \"GS12_15FU1U2U3_OPS\":GS12_15FU1U2U3_OPS,\n",
    "        \n",
    "        \"GS12_25S\":GS12_25S,\n",
    "        \"GS12_25FS\":GS12_25FS,\n",
    "        \"GS12_25FU1S\":GS12_25FU1S,\n",
    "        #\"GS12_25FU1PATCHS\":GS12_25FU1PATCHS,\n",
    "        #\"GS12_25FU1U2U3S\":GS12_25FU1U2U3S,\n",
    "        \"GS12_25FU1U2U3_OPS\":GS12_25FU1U2U3_OPS,\n",
    "        \n",
    "        \n",
    "        \"GS12_30S\":GS12_30S,\n",
    "        \"GS12_30FS\":GS12_30FS,\n",
    "        \"GS12_30FU1S\":GS12_30FU1S,\n",
    "        #\"GS12_30FU1PATCHS\":GS12_30FU1PATCHS,\n",
    "        #\"GS12_30FU1U2U3S\":GS12_30FU1U2U3S,\n",
    "        \"GS12_30FU1U2U3_OPS\":GS12_30FU1U2U3_OPS,\n",
    "        \n",
    "        \n",
    "        \"GS12_50S\":GS12_50S,\n",
    "        \"GS12_50FS\":GS12_50FS,\n",
    "        \"GS12_50FU1S\":GS12_50FU1S,\n",
    "        #\"GS12_50FU1PATCHS\":GS12_50FU1PATCHS,\n",
    "        #\"GS12_50FU1U2U3S\":GS12_50FU1U2U3S,\n",
    "        \"GS12_50FU1U2U3_OPS\":GS12_50FU1U2U3_OPS,\n",
    "\n",
    "\n",
    "        \"jy0112_1S\":jy0112_1S,\n",
    "        \"jy0112_1FS\":jy0112_1FS,\n",
    "        \"jy0112_1FU1S\":jy0112_1FU1S,\n",
    "        #\"jy0112_1FU1PATCHS\":jy0112_1FU1PATCHS,\n",
    "        #\"jy0112_1FU1U2U3S\":jy0112_1FU1U2U3S,\n",
    "        \"jy0112_1FU1U2U3_OPS\":jy0112_1FU1U2U3_OPS,\n",
    "        \n",
    "        \"jy005S\":jy005S,\n",
    "        \"jy005FS\":jy005FS,\n",
    "        \"jy005FU1S\":jy005FU1S,\n",
    "        #\"jy005FU1PATCHS\":jy005FU1PATCHS,\n",
    "        #\"jy005FU1U2U3S\":jy005FU1U2U3S,\n",
    "        \"jy005FU1U2U3_OPS\":jy005FU1U2U3_OPS,\n",
    "        \n",
    "     \n",
    "    }\n",
    "    dic_ssim_patchs={\n",
    "        #\"GS12_15S\":GS12_15S,\n",
    "        #\"GS12_15FS\":GS12_15FS,\n",
    "        #\"GS12_15FU1S\":GS12_15FU1S,\n",
    "        \"GS12_15FU1PATCHS\":GS12_15FU1PATCHS,\n",
    "        \"GS12_15FU1U2U3S\":GS12_15FU1U2U3S,\n",
    "        #\"GS12_15FU1U2U3_OPS\":GS12_15FU1U2U3_OPS,\n",
    "        \n",
    "        #\"GS12_25S\":GS12_25S,\n",
    "        #\"GS12_25FS\":GS12_25FS,\n",
    "        #\"GS12_25FU1S\":GS12_25FU1S,\n",
    "        \"GS12_25FU1PATCHS\":GS12_25FU1PATCHS,\n",
    "        \"GS12_25FU1U2U3S\":GS12_25FU1U2U3S,\n",
    "        #\"GS12_25FU1U2U3_OPS\":GS12_25FU1U2U3_OPS,\n",
    "        \n",
    "        \n",
    "        #\"GS12_30S\":GS12_30S,\n",
    "        #\"GS12_30FS\":GS12_30FS,\n",
    "        #\"GS12_30FU1S\":GS12_30FU1S,\n",
    "        \"GS12_30FU1PATCHS\":GS12_30FU1PATCHS,\n",
    "        \"GS12_30FU1U2U3S\":GS12_30FU1U2U3S,\n",
    "        #\"GS12_30FU1U2U3_OPS\":GS12_30FU1U2U3_OPS,\n",
    "        \n",
    "        \n",
    "        #\"GS12_50S\":GS12_50S,\n",
    "        #\"GS12_50FS\":GS12_50FS,\n",
    "        #\"GS12_50FU1S\":GS12_50FU1S,\n",
    "        \"GS12_50FU1PATCHS\":GS12_50FU1PATCHS,\n",
    "        \"GS12_50FU1U2U3S\":GS12_50FU1U2U3S,\n",
    "        #\"GS12_50FU1U2U3_OPS\":GS12_50FU1U2U3_OPS,\n",
    "\n",
    "\n",
    "        #\"jy0112_1S\":jy0112_1S,\n",
    "        #\"jy0112_1FS\":jy0112_1FS,\n",
    "        #\"jy0112_1FU1S\":jy0112_1FU1S,\n",
    "        \"jy0112_1FU1PATCHS\":jy0112_1FU1PATCHS,\n",
    "        \"jy0112_1FU1U2U3S\":jy0112_1FU1U2U3S,\n",
    "        #\"jy0112_1FU1U2U3_OPS\":jy0112_1FU1U2U3_OPS,\n",
    "        \n",
    "        #\"jy005S\":jy005S,\n",
    "        #\"jy005FS\":jy005FS,\n",
    "        #\"jy005FU1S\":jy005FU1S,\n",
    "        \"jy005FU1PATCHS\":jy005FU1PATCHS,\n",
    "        \"jy005FU1U2U3S\":jy005FU1U2U3S,\n",
    "        #\"jy005FU1U2U3_OPS\":jy005FU1U2U3_OPS,\n",
    "        \n",
    "     \n",
    "    }\n",
    "       \n",
    "        \n",
    "                \n",
    "         \n",
    "    df_psnr_IMAGE=pd.DataFrame(dic_psnr_images)\n",
    "    df_ssim_IMAGE=pd.DataFrame(dic_ssim_images)\n",
    "    df_psnr_PATCH=pd.DataFrame(dic_psnr_patchs)\n",
    "    df_ssim_PATCH=pd.DataFrame(dic_ssim_patchs)\n",
    "    return df_psnr_IMAGE,df_ssim_IMAGE,df_psnr_PATCH,df_ssim_PATCH\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\"\"\"def cc(df,m):\n",
    "    name = {\"Airplane55\":df.loc[55][m],\"Barbara86\":df.loc[86][m],\"Boats1\":df.loc[1][m],\"C.man0\":df.loc[0][m],\"Couple3\":df.loc[3][m],\"House11\":df.loc[11][m],\"Lena77\":df.loc[77][m],\"Man2\":df.loc[55][m],\"Monarch44\":df.loc[44][m],\"Parrot66\":df.loc[66][m],\"Peppers22\":df.loc[22][m],\"Starfish33\":df.loc[33][m]}\n",
    "    mean = (sum(name.values()))/12\n",
    "    return name,mean\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ba0994b",
   "metadata": {},
   "outputs": [],
   "source": [
    "sahpe1,sahpe2 = [256,256]\n",
    "df_psnr_IMAGE,df_ssim_IMAGE,df_psnr_PATCH,df_ssim_PATCH = com_df()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42c66a31",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_psnr_IMAGE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e44d3356",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_psnr_IMAGE.to_csv(\"df_psnr_IMAGE.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff7184a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_ssim_IMAGE.to_csv(\"df_ssim_IMAGE.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08b4cd2b",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "df_psnr_PATCH.to_csv(\"df_psnr_PATCH.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c60b8b26",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "df_ssim_PATCH.to_csv(\"df_ssim_PATCH.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fe9acdf8-570a-46dd-9355-d5211638903d",
   "metadata": {},
   "outputs": [],
   "source": [
    "sahep_ls=[32,64,128,256,512,1024]\n",
    "m = 0\n",
    "max_mean0=[]\n",
    "for sahpe1 in sahep_ls:\n",
    "    for sahpe2 in sahep_ls:\n",
    "        df = com_df()\n",
    "        name,mean = cc(df,m)\n",
    "        max_mean0.append([mean,[name],sahpe1,sahpe2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "adf5b897-bfaf-4769-93d3-947237b5b72e",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(max_mean0)):\n",
    "    print(max_mean0[i][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f53fdfe-37ce-4350-ad2a-76d907c6f560",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2fcf3b0-ac17-4537-a59b-55d7fdf4f2e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "m = 1\n",
    "max_mean1=[]\n",
    "for sahpe1 in sahep_ls:\n",
    "    for sahpe2 in sahep_ls:\n",
    "        df = com_df()\n",
    "        name,mean = cc(df,m)\n",
    "        max_mean1.append([mean,[name],sahpe1,sahpe2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9b6ca715-51db-40e0-8dd6-1b5a2e80b3eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(max_mean1)):\n",
    "    print(max_mean1[i][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf9fe0f6-4e6d-4ba1-9011-99208124544e",
   "metadata": {},
   "outputs": [],
   "source": [
    "max_mean1[8]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "74b480bc-db70-4316-8d92-4088c9c3f49d",
   "metadata": {},
   "outputs": [],
   "source": [
    "m = 2\n",
    "max_mean2=[]\n",
    "for sahpe1 in sahep_ls:\n",
    "    for sahpe2 in sahep_ls:\n",
    "        df = com_df()\n",
    "        name,mean = cc(df,m)\n",
    "        max_mean2.append([mean,[name],sahpe1,sahpe2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10d658e8-5a11-4ab7-8856-f23cb442585f",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(max_mean2)):\n",
    "    print(max_mean2[i][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7c0c7c1-8175-4f44-aa1c-762df5751017",
   "metadata": {},
   "outputs": [],
   "source": [
    "m = 3\n",
    "max_mean3=[]\n",
    "for sahpe1 in sahep_ls:\n",
    "    for sahpe2 in sahep_ls:\n",
    "        df = com_df()\n",
    "        name,mean = cc(df,m)\n",
    "        max_mean3.append([mean,[name],sahpe1,sahpe2])\n",
    "for i in range(len(max_mean3)):\n",
    "    print(max_mean3[i][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d17a6949-2c6e-4d30-8555-13a257bd147b",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_dic= {\"max_mean25\":max_mean1[8],\"max_mean50\":max_mean3[8]}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "65df8fb4-9096-4cc8-ade8-c8c3579d34aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_dic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f307782",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_csv(\"DN_TEST.csv \")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62f112f6",
   "metadata": {},
   "source": [
    "# 语义分割"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9bd021f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pixellib\n",
    "import glob\n",
    "from pixellib.semantic import semantic_segmentation\n",
    "segment_image = semantic_segmentation()\n",
    "segment_image.load_pascalvoc_model(r\"C:\\Users\\Administrator\\.keras\\models\\deeplabv3_xception_tf_dim_ordering_tf_kernels.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45eab72f",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_path = glob.glob(r\"D:\\Research Related\\Paper Related\\Step-by-step image generation\\Code\\WANGHONG\\PATH\\to\\SAGAN_DONGMAN\\*.jpg\")\n",
    "save_path = r\"D:\\Research Related\\Paper Related\\Step-by-step image generation\\Code\\dataset\\cartoon512seg/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "380b2d98",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_path[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a83d4d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in img_path:\n",
    "    segment_image.segmentAsPascalvoc(i, output_image_name = save_path+str(i.split(\"\\\\\")[7].split(\".\")[0])+\".jpg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3682ba1f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.11"
  },
  "nbTranslate": {
   "displayLangs": [
    "*"
   ],
   "hotkey": "alt-t",
   "langInMainMenu": true,
   "sourceLang": "en",
   "targetLang": "fr",
   "useGoogleTranslate": true
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": true,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": true,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
