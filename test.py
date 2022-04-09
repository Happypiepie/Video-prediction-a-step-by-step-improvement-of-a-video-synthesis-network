{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import os\n",
    "import glob\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "%matplotlib inline\n",
    "import time\n",
    "from IPython import display"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "imgs_path1 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise15_F\\GSnoise15_F_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path1.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "imgs_path2 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise25_F\\GSnoise25_F_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path2.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "imgs_path3 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise30_F\\GSnoise30_F_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path3.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "imgs_path4 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise50_F\\GSnoise50_F_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path4.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "imgs_path5 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.1_M_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path5.sort(key=lambda x:int(x.split('\\\\')[7].split('.')[0]))\n",
    "imgs_path6 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.05_M_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path6.sort(key=lambda x:int(x.split('\\\\')[7].split('.')[0]))\n",
    "imgs_pathy = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise15_F\\test_SET_ORG640_480\\*.jpg')\n",
    "imgs_pathy.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'F:\\\\Videodenoise\\\\dataset\\\\test\\\\GS\\\\GS_NOISE_F\\\\GSnoise15_F\\\\GSnoise15_F_U1U2HIGHout_low_outIMAGES\\\\5.jpg'"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "imgs_path1[5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "channel,shape1,shape2,bn = [3,480,640,16]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_and_preprocess1(path):\n",
    "    img = tf.io.read_file(path)\n",
    "    img = tf.image.decode_png(img,channels=channel)\n",
    "    img = tf.image.resize(img,[shape1,shape2])    \n",
    "    img = (tf.cast(img,tf.float32)-127.5)/127.5    \n",
    "    return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_x1 = tf.data.Dataset.from_tensor_slices(imgs_path1).map(load_and_preprocess1).batch(bn)\n",
    "dataset_x2 = tf.data.Dataset.from_tensor_slices(imgs_path2).map(load_and_preprocess1).batch(bn)\n",
    "dataset_x3 = tf.data.Dataset.from_tensor_slices(imgs_path3).map(load_and_preprocess1).batch(bn)\n",
    "dataset_x4 = tf.data.Dataset.from_tensor_slices(imgs_path4).map(load_and_preprocess1).batch(bn)\n",
    "dataset_x5 = tf.data.Dataset.from_tensor_slices(imgs_path5).map(load_and_preprocess1).batch(bn)\n",
    "dataset_x6 = tf.data.Dataset.from_tensor_slices(imgs_path6).map(load_and_preprocess1).batch(bn)\n",
    "dataset_y = tf.data.Dataset.from_tensor_slices(imgs_pathy).map(load_and_preprocess1).batch(bn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset1 = tf.data.Dataset.zip((dataset_x1,dataset_y)).shuffle(16)\n",
    "dataset2 = tf.data.Dataset.zip((dataset_x2,dataset_y)).shuffle(16)\n",
    "dataset3 = tf.data.Dataset.zip((dataset_x3,dataset_y)).shuffle(16)\n",
    "dataset4 = tf.data.Dataset.zip((dataset_x4,dataset_y)).shuffle(16)\n",
    "dataset5 = tf.data.Dataset.zip((dataset_x5,dataset_y)).shuffle(16)\n",
    "dataset6 = tf.data.Dataset.zip((dataset_x6,dataset_y)).shuffle(16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<ShuffleDataset shapes: ((None, 480, 640, 3), (None, 480, 640, 3)), types: (tf.float32, tf.float32)>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def downsample(filters, size, apply_batchnorm=True):\n",
    "#    initializer = tf.random_normal_initializer(0., 0.02)\n",
    "\n",
    "    result = tf.keras.Sequential()\n",
    "    result.add(\n",
    "        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',\n",
    "                               use_bias=False))\n",
    "\n",
    "    if apply_batchnorm:\n",
    "        result.add(tf.keras.layers.BatchNormalization())\n",
    "\n",
    "        result.add(tf.keras.layers.LeakyReLU())\n",
    "\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def upsample(filters, size, apply_dropout=False):\n",
    "#    initializer = tf.random_normal_initializer(0., 0.02)\n",
    "\n",
    "    result = tf.keras.Sequential()\n",
    "    result.add(\n",
    "        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,\n",
    "                                        padding='same',\n",
    "                                        use_bias=False))\n",
    "\n",
    "    result.add(tf.keras.layers.BatchNormalization())\n",
    "\n",
    "    if apply_dropout:\n",
    "        result.add(tf.keras.layers.Dropout(0.5))\n",
    "\n",
    "    result.add(tf.keras.layers.ReLU())\n",
    "\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Generator():\n",
    "    inputs = tf.keras.layers.Input(shape=[480,640,3])\n",
    "\n",
    "    down_stack = [\n",
    "        downsample(64, 3), # (bs, 16, 16, 64)\n",
    "        downsample(128, 3), # (bs, 8, 8, 128)\n",
    "        downsample(256, 3), # (bs, 4, 4, 256)\n",
    "        downsample(512, 3), # (bs, 2, 2, 512)\n",
    "        downsample(512, 3), # (bs, 1, 1, 512)\n",
    "    ]\n",
    "\n",
    "    up_stack = [\n",
    "        upsample(512, 3, apply_dropout=True), # (bs, 2, 2, 1024)\n",
    "        upsample(256, 3, apply_dropout=True), # (bs, 4, 4, 512)\n",
    "        upsample(128, 3, apply_dropout=True), # (bs, 8, 8, 256)\n",
    "        upsample(64, 3), # (bs, 16, 16, 128)\n",
    "        \n",
    "    ]\n",
    "\n",
    "#    initializer = tf.random_normal_initializer(0., 0.02)\n",
    "    last = tf.keras.layers.Conv2DTranspose(3, 3,\n",
    "                                         strides=2,\n",
    "                                         padding='same',\n",
    "                                         activation='tanh') # (bs, 64, 64, 3)\n",
    "\n",
    "    x = inputs\n",
    "\n",
    "    # Downsampling through the model\n",
    "    skips = []\n",
    "    for down in down_stack:\n",
    "        x = down(x)\n",
    "        skips.append(x)\n",
    "\n",
    "    skips = reversed(skips[:-1])\n",
    "\n",
    "    # Upsampling and establishing the skip connections\n",
    "    for up, skip in zip(up_stack, skips):\n",
    "        x = up(x)\n",
    "        x = tf.keras.layers.Concatenate()([x, skip])\n",
    "\n",
    "    x = last(x)\n",
    "\n",
    "    return tf.keras.Model(inputs=inputs, outputs=x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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c4MoOrxm8TKORFmdltC9jivV9t36cOs7lV3vXsZC8egNy9wAAESUqdfRhPFcYtCFmp99MCvpjmISAIA4fr4r0g8mipgwXm0UND/7oJB0xzI3AABxKByA5DAzCQAAgJRRTAIAACBlsVrm8wFkmUWLFunCCy9Up06dTEeBz/bs2aPy8nLFYjG1aNHCdJy0bNu2TXl5eSoqKvJ0/p49e1RZWalmzZoFnAyp+uCDD9S1a1ctWbLEdJQwjaGYBJB1ampq9M9//pNn37LMihUrNGXKFN15553q0qWL6Tih+/LLL3X99dfr4IMP1uTJk9W0aVPTkWCjdevWatKkiekYYaKYBABE3z333KO5c+dq4cKFOvzww03HMaampkbTp0/XzJkz9eCDD+rMM880HQmgmAQARFdlZaWGDx+uyspKPfbYY56XhLPdBx98oMsuu0ydO3fWPffco4KCAtORkLvGsAEHABBJn3/+uXr06KEjjzxSCxcupJCM89Of/lSvvfaaWrZsqa5du2rDhg2mIyGHMTMJAIiclStX6oorrtC9996rn//856bjRNry5ct11VVXaeLEibryyitNx0HuYZkbABAts2bN0vTp07VgwQIdffTRpuNkhG+//VbDhg1T06ZNNWvWLBUXF5uOhNzBMjcAIBr27dunq6++WosWLdLq1aspJJPQunVrPfvsszrppJPUtWtXvf/++6YjIYdQTAIAjPvmm2/Uu3dvFRYW6rnnnlPz5s1NR8o4sVhM119/vWbMmKHzzz9fixYtMh0JOYJlbgCAUevWrdMll1yiyZMna+jQoabjZIXPP/9cAwYM0DnnnKMpU6aoXj3mjhAYlrkBAObMmzdPl1xyiebNm0ch6aNDDjlEK1as0Geffab+/furrKzMdCRkMYpJAEDoqqurdcMNN2jGjBlauXKlTj75ZNORsk7jxo31xz/+Ub169dIZZ5yhL774wnQkZCmKSQBAqL7//nudd9552rFjh5YtW6YDDzzQdKSsdu2112ry5Mnq1auX3n33XdNxkIUoJgEAodmwYYO6deum/v37a/bs2crLyzMdKSf0799fjz32mC688EItW7bMdBxkGYpJAEAonn76aV1wwQWaPXu2rr76atNxcs6pp56qF154QWPHjtXcuXNNx0EWaWA6AAAgu9XW1urWW2/V4sWLtXz5ch1yyCGmI+WsI444Qq+88orOO+88VVRUaMSIEaYjIQtQTAIAAlNeXq7LLrtM+fn5WrlypfLz801HynmtW7fWSy+9pHPPPVd79+7V6NGjTUdChmOZGwAQiI8//ljdu3fXaaedprlz51JIRkjTpk314osv6sknn9S9995rOg4yHDOTAADfvfTSS7rmmmv00EMP6eyzzzYdBzaaNGmiF154QRdccIH27t2rG2+80XQkZCiKSQCAr6ZNm6Y5c+boxRdf1BFHHGE6DlwUFBTomWee0QUXXKD8/Hxde+21piMhA1FMAgB8sXv3bo0YMUI7d+7U6tWr1aRJE9OR4EF+fr6efvpp9e7dW61bt9aQIUNMR0KG4ZlJAEDaNm/erNNPP10dOnTQ//7v/1JIZpiCggItXrxYd999t55//nnTcZBhmJkEAKRl9erVuvzyy3XPPffowgsvNB0HKWrdurWee+459e7dWy1bttSpp55qOhIyBDOTAICU/c///I9GjBihRYsWUUhmgXbt2mnBggW64oortGnTJtNxkCGYmQQAJG3fvn0aN26cPv74Y61atUotWrQwHQk+Oe644/TAAw9owIABWrVqlYqLi01HQsQxMwkAcHX++efrrLPO2v/7b775Rr1791ZBQYFeeOEFCsks1Lt3b11xxRUaNmyYamtrTcdBxFFMAgAcrVu3TitWrNAbb7yh6dOna926derevbtGjhype++9V/Xr1zcdEQG57rrrVFhYqFtvvdV0FERcrJb/5AAA2KiurtYxxxyjjRs3SpKKi4vVsmVLPfnkkzr55JMNp0MYKioqdPrpp+uWW25Rv379TMdBNI3hmUkAgK0HHnhAX3755f7fl5WVqWHDhjrooIMMpkKYCgoK9Je//EU9e/ZUx44d1bZtW9OREEHMTAIAfuSrr77S0Ucfre3bt//geL169VRSUqJ33nmHjRk5ZP78+frjH/+oF154QbFYzHQcRMsYnpkEAPzI8OHDVV5e/qPjTZs21WeffbZ/6Ru54dJLL1XTpk01a9Ys01EQQRSTAIAf+Otf/6rVq1erqqpK0r+elWzatKkGDBigBQsWqLq6Wp07dzacEmGbOXOmpk2bpo8++sh0FEQMy9wAgP12796t/Px85eXlKT8/X8cff7zGjh2rfv36KT8/33Q8GPb888/rtttu08qVK9nJjzpjKCYBH91555369a9/rQ4dOpiO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b2D8eXYbLtyu96MIsc6AeWnbbexOee3GmWjZO9HYne6j1z7s2ndqJ5X2/RpD3bmmZ2J79uypV1991WgG+I9iEgAMSrRs6/bVjnbnJXt9UM8LOmX0ep5bXqfNRV6zWO+J02de+vA6hlTb92sMUdGxY0etW7fOdAz4jGISAJA2U8/nhTHbFnQf2TAGr0466SSKySxEMQkAGcrPpWo/mChWwugzjNcnBS0KhaQktWvXTl999ZWqq6tNR4GPKCYBIENFeTkTcPKTn/xEH374oekY8BHFJAAACM2JJ56od955x3QM+IhiEgAAhKZDhw765JNPTMeAjxqYDgBkk+LiYg0ZMkS/+c1vTEdBBNTU1KiiokKHH374D45XVlaqXr16atSokaFkydm7d69qamr05JNPmo4i6V/L+1u3blVxcXHG3MNM8sknn6i4uDiw9jt06KCnnnoqsPYRvlgtD9sAQGhKS0vVs2dPvfzyy2rfvr3pOBlryZIlmjJlilq0aKFbbrlFnTt3Nh0JHm3atEkjR47U8uXLTUeBP8awzA0AIamqqtKQIUP029/+lkIyTWeddZZef/11jR49WmPGjNEFF1zAK2cyxKGHHqrPP//cdAz4iGISAEIyefJknXDCCbrwwgtNR8ka5557rt58802NGDFCI0eO1IABA/i6vojLy8tTVVWVqqqqTEeBTygmASAES5Ys0ZIlS3TvvfeajpKVzj//fK1Zs0YXX3yxzjnnHF1//fXavn276Vhw0KpVK23bts10DPiEYhIAArZlyxaNHj1ac+fOZcNIgGKxmC699FL94x//UPPmzdWxY0fNmDGDGbAIatGihb777jvTMeATikkACFB1dbWGDh2qW265RUceeaTpODkhPz9fN910k1577TW9++676tSpk5YsWWI6FuK0atVK3377rekY8AnFJAAE6K677tKhhx6qIUOGmI6Scw4++GDNmDFDc+fO1a233qpLL71UW7ZsMR0L+lcxycxk9qCYBICArFixQk888YQeeOAB01Fy2rHHHqsVK1aoZ8+e6tq1q2bPnq2amhrTsXJay5YteWYyi1BMAkAAtm7dqhEjRmj+/PkqKCgwHSfnxWIxjRw5Uq+//rpWrlyp008/XevXrzcdK2fl5+ervLzcdAz4hGISAHxWW1urYcOGacKECTr22GNNx0GcAw88UI8//rimTJmigQMH6vbbb1d1dbXpWDmnSZMmqqysNB0DPqGYBACfTZs2TcXFxRo5cqTpKHBw5plnau3atSotLdXpp5+ujz/+2HSknNKoUSPt2rXLdAz4hGISAHz0t7/9TY888ohmzZplOgoSKCws1OzZs3XjjTeqb9++evjhh01HyhmNGzfWvn37TMeATygmAcAn27dv12WXXaY//elPKjkdZVQAACAASURBVC4uNh0HHvXr10+rV6/W4sWL1b9/f15ZE4KioiKVlZWZjgGfUEwCgE9GjBihUaNG6aSTTjIdBUk64IADtHjxYp133nnq1q2bXn/9ddORslr9+vVNR4CPKCYBwAczZ87Uvn37NHbsWNNRkIYRI0ZowYIFGj58uGbOnGk6DpARKCYBIE3vvPOOfve73+mRRx5RLBYzHQdpOv7447V69Wo9++yz+uUvf6ndu3ebjgREGsUkAKShvLxcQ4YM0SOPPKIWLVqYjgOfNGvWTM8884xKSkrUo0cP/fOf/zQdCYgsikkASMOoUaM0dOhQde3a1XQU+KxevXq65ZZbNGXKFJ155pl68803TUcCIoliEgBSNGfOHH399deaOHGi6SgIUL9+/fSXv/xFl19+uZ5//nnTcbIC75jMLg1MBwCATPTBBx/o9ttv1+rVq3lOMgcce+yxeumll9SvXz998803GjZsmOlIGa2qqkoNGzY0HQM+YWYSAJJUWVmpSy+9VLNnz9YBBxxgOg5Ccsghh+iVV17RI488ottvv910nIzHd9ZnD4pJAEjSddddp/PPP1+9evUyHQUha968uf7617/q73//u8aMGaPa2lrTkQDjKCYBwMX27dtVVVW1//d//vOf9f7772vKlCkGU8Gk/Px8LViwQGVlZRo7diwFJXIexSQAuGjevLk6dOig0tJSffLJJ7rppps0f/58vsEjx9WvX1+PPvqovvnmG11//fWm4wBGUUwCgIOtW7eqadOm+uKLL9SxY0f169dP06dPV5s2bUxHQwTUr19fjz/+uD7++GPdfPPNpuMAxlBMAoCDZ599VlVVVaqpqdG2bdu0efNmLVq0SHv27DEdDRGRl5enJ598UmvXrtWtt95qOk7G2LVrl/Ly8kzHgE8oJgHAwZw5c37wPrydO3dq7ty5ys/P1/bt2w0mQ5Tk5eVp4cKFWr58uaZPn246TkaoqqpSfn6+6RjwCcUk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9tln1bBhQ9NxMlpFRYVmzZqlhx56SBdccIFuuOEGHXzwwaZjwQHL3Ig6npkEkBHGjx+vu+++m0LSBwUFBfqv//ovvfvuu2rfvr169OihcePGafPmzaajwcbBBx+sLVu2mI4BOKKYBBB5ixYtUn5+vs455xzTUbJKfn6+xo8fr3fffVdHHnmkfvazn+nGG2/U9u3bTUdDnKZNm/JngkijmAQQabt379bEiRN13333mY6StRo3bqzRo0frH//4h4qKitSxY0fdf//92rt3r+lokNSwYUP+LBBpFJMAIm3atGk6//zz9ZOf/MR0lKxXWFiom266SX/729+0adMmnXjiiZo/fz4vegfgimISQGR98cUXevTRRzV58mTTUXJK69at9dBDD2nx4sVasGCBunXrprfeest0rJzGq5wQZRSTACJr0qRJmjRpkoqLi01HyUlHHHGEFi5cqNtuu02XXXaZxo8frx07dpiOlZMKCwtVVlZmOgZgi2ISQCS98cYb2rhxo6644grTUXJer169tHbtWh100EE66aSTNH/+fNORck6DBg143ACRRTEJIHJqamo0btw43X///apXj/83FQV5eXm68cYb9fLLL2v+/Pnq06ePNm3aZDoWgAjg/0sDiJzHHntMP/nJT9S1a1fTUWDRvn17LV68WOPGjdN5552nBx54gBkzIMdRTAKIlLKyMt1+++26++67TUeBiwsuuEBvvPGGXnvtNfXp00elpaWmIwEwhGISQKTUbfZo27at6ShIoEWLFpo/f76uvvpq9erVS3/84x9NRwJgAMUkgMjYtGmTnnnmGf3qV78yHQVJGDhwoFatWqWnnnpK/fv313fffWc6EoAQUUwCiIzrrrtOt912mxo3bmw6CpJ00EEH6dlnn9U555yjrl27as2aNaYjZZWdO3eqoKDAdAzAFsUkgEhYsmSJKioqNGDAANNRkIarrrpKTzzxhC677DLNnDnTdJyssW/fPjVs2NB0DMAWxSQA4/bt26cJEyZo+vTppqPABx07dtTrr7+u559/XkOHDlVFRYXpSBmPb8BBlFFMAjDuoYceUo8ePXTssceajgKfNGvWTIsXL9bRRx+trl278k7KNFVVVZmOADhqYDoAgNz27bff6sEHH9Sbb75pOgp8FovF9Otf/1pdunTRueeeqzlz5qhbt26mY2Wc6upqNWjAP9eILmYmARh10003afz48WrZsqXpKAhIr169tHjxYg0fPlwLFy40HSfjlJeXq2nTpqZjAI74Tx0Axrz99tv629/+phkzZpiOgoAdddRRWr58uc4//3xt3rxZ48ePNx0pY1RUVKhRo0amYwCOmJkEYERtba3GjRunadOmsYSXIw466CAtX75cL774ov7rv/5LNTU1piNlhD179vC6LEQaxSQAI5566im1bNlSvXr1Mh0FISoqKtIzzzyjHTt2aMiQIaqurjYdKfJ27typ4uJi0zEARxSTAEIxa9YszZo1S7W1taqoqNBvfvMbTZs2zXQsGNCgQQM9/PDDatWqlf7zP/+TgjKByspK5eXlmY4BOKKYBBCK22+/Xdddd52OOeYYjR8/XgMGDNBhhx1mOhYMicVimj59upo2baorrriCgtLFnj17+PYbRBrFJIDA1dbWaseOHdq9e7fef/99PfHEE3r//ff11VdfmY4Gg2KxmGbMmKG8vDyNGDGCZygdlJWVqUmTJqZjAI4oJgEE7pNPPvnBV8GVl5fr2WefVZs2bTR79myDyWBaLBbT73//e0nSyJEjVVtbazhR9OzcuVNFRUWmYwCOKCYBBO7tt9/+0Td4VFdXKz8/X4cffrihVIiKuoJy9+7d+tWvfmU6TuSUl5czM4lIo5gEELjXX39dZWVl+3/foEEDtWnTRuvWrVOfPn0MJkNU1K9fX4888ojeeustzZw503ScSCkvL2dmEpHGy90ABG7lypX7f66bjXz55Zd1wAEHGEyFqMnLy9OCBQvUvXt3HXnkkbw26t/Ky8vVunVr0zEAR8xMAgjcpk2bJElNmjRRr169tGbNGgpJ2GrevLmefvppDR8+fP/fm1zHMjeijmISQKC+/vpr7dmzR8XFxbryyiv1zDPP8G0ecHXEEUfo4YcfVv/+/bV161bTcYwrKyvjpeWINJa5c8jXX3+tFStWmI6BHPPmm2+qoqJCV111lbp27aoFCxaYjoQIOeyww3TyySf/6HivXr107bXX6tJLL9WLL76o+vXrG0gXDWVlZTwziUijmMwhf/zjH3XHHXfo6quvNh0FETF37lwNGTIk0D727NmjIUOGqFmzZlq7dq3v7W/ZskXvvfeeevfu7XvbCNaXX36ppUuX6ssvv7T9fMSIEXrzzTd1xx136De/+U3I6aKDDTiIOorJHFJTU6NJ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CrK0du1adezYUd27d9fQoUMpJAEg5FgziViqUaOGxo4dq6KiIp155plq1aqV6Uhw4YsvvtD111+v5557TpdffrnpOAAAFxiZRGw1bdpUf/7zn3XDDTeopKTEdBxk8Ne//lXXX3+9Xn/9dQpJAIgQiknEWlFRkW699VbddtttpqPAxv79+zVgwAC98sormjNnjtq0aWM6EgAgCxSTiL17771XFRUVevHFF01HQZp169apU6dOqlOnjt5//301adLEdCQAQJYoJhF7iURCf/7znzVq1CjNnz/fdBz8x8SJE9WxY0fdc889GjZsmAoKWMINAFFEMYm80KBBA40dO1Y333yztmzZon379unaa69VvXr1TEfLO7t379btt9+u5557TjNmzNC1115rOhIAoBooJpE3zjjjDD300EPq0aOHTjvtNE2ZMkU1atTQxo0bTUeLnR07dqiiouKg7QsWLFC7du10/PHH68MPP9Rxxx1nIB0AwEsUk8grRx55pBYvXqwVK1aotLRU+/fv19/+9jfTsWJl8+bNatiwoR555JEftpWWlur+++/XbbfdpjFjxuiRRx5RjRo1DKYEAHiFYhJ5o3///rrxxhu1Y8cOJZNJSdKePXv08ssvG04WH6WlpbrgggtUUFCgUaNGacOGDfrggw/Upk0b1atXTwsXLlS7du1MxwQAeIgV78gLe/bs0UsvvaRDDz30oNdWrlyprVu36vDDDzeQLD4qKip0+eWXa9WqVSovL9fu3bt16aWXqm7dunrrrbd4cDwAxBQjk8gLderUUVlZme6++241aNDggNf279+viRMnGkoWH3379tVnn32mvXv3SpLKy8u1Zs0ajRgxgkISAGKMYhJ5o7CwUL/5zW80e/ZsnXjiiT/cyV1aWqrRo0cbThdtw4YN08SJE7V79+4DtpeUlKhfv36GUgEAgkAxibzTqlUrLV26VA899JAaNmyoRCKh5cuXa/v27aajRdLEiRP11FNPWX5lZa1atfT5559rwYIFBpIBAIJAMYm8VFBQoMcee0zz58/XqaeeqpKSEr366qumY0XOjBkzdPXVV6ukpEQNGjRQ48aNVb9+fZ1yyim66aabNHz4cM2fP19nnXWW6agAAJ8ccANOIpFQixYtTGVBxOzZs0fSv9cjRtmuXbtUr149/frXv9bw4cNNx4mUdevWqaCgQIcddphq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9erVuuuuu1SjRg3NmjVLxx57rOlIALLEyCQA33Tt2lWTJ082HQMhVFlZqRdeeEEXX3yx+vTpo3fffZdCEogoRiYB+KZZs2ZKJBJav34905b4wRdffKH+/fvrpJNO0qeffqrGjRubjgSgGhiZBOCrrl27aurUqaZjIAS2bNmifv36qU+fPho6dKheffVVCkkgBigmAfiqat0k8ld5eblefPFFtWvXTqeddpq++OIL1tECMUIxCcBXRUVFmj9/vioqKkxHgQEffvihWrduraVLl+qzzz7T3XffrYICVlgBccLfaAC+KiwsVJs2bTRv3jwVFRWZjoOAfP3113rkkUe0adMm/eUvf1Hr1q1NRwLgE0YmAfiuS5curJvME8XFxerbt6+uu+469enTR3PnzqWQBGKOYhKA7zp37qxp06aZjgEfbdmyRQ888IAuvvhinXvuufriiy/UvXt3JRIJ09EA+IxiEoDvmjdvrv3792vDhg2mo8BjO3fu1FNPPaWzzjpLTZs21ZdffqnbbruNdZFAHqGYBBAI7uqOl5KSEj311FM67bTTVFpaqs8//1wPPfSQ6tSpYzoagIBRTAIIRJcuXZjqjoEdO3boiSee0Omnn659+/Zp0aJFeuqpp9SwYUPT0QAYQjEJIBBFRUX65JNPVFFRodLSUk2cOFHbt283HQsuff/99/rVr36lM844Q8lkUp9//rmeeOIJHjoOgGISQDC+/fZbNW/eXK1bt9bRRx+tq6++WnPnzjUdCxmsWbNG9957r9q0aaPCwkItXrxYv/rVr9SoUSPT0QCEBCukAfhqzJgxuuWWW3T44YerpKRE+/fvlyQ1btxYDRo0MJwOdhYtWqRnn31Wixcv1j333KOlS5eqbt26pmMBCCGKSQC+atmyperUqaOtW7cesD2RSLDOLmSSyaSmT5+uZ555Rnv27NGDDz6osWPH6pBDmMQCYI9/IQD4qk2bNnryySdVv379g15jZDI4e/bsUZs2bXT88ccf9NrOnTv1hz/8QS1bttRLL72kX//615o3b566d+9OIQkgI/6VAOC7++67TyeffLJq1Kjxw7aKigqKyYBs2LBBbdq00ddff63t27dr48aNkqTly5drwIABOu2007RmzRq99957mjhxojp06GA4MYAoSSSTyaTpEADi77vvvlPLli21bds2SVK9evW0c+dOw6ni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1DEdB8grbdq00bRp09SkSRPTUeA/RiYBxFOzZs10xRVXMDoJBGzu3Ln6r//6LwrJPMLIJIDY2rBhg84991wtW7aM0UkgIH369FHfvn11wQUXmI6CYDAyCSC+jjrqKF1zzTUaMWKE6ShAXvjuu+/0zTffUEjmGUYmAcTapk2b1L59ey1ZskSHHnqopkyZolq1aumiiy4yHQ2Inccee0w/+clP1KdPH9NREJwBFJMAYu/hhx/W9u3b9cknn+if//yn/uu//ktff/216VhArOzZs0dnn322Fi5cyIPK88uAAtMJAMBPH3/8saZMmaI1a9Zo586dkqTS0lLDqYD4+ctf/qKePXtSSOYhikkAsdWyZUv961//0r59+w7Yvm3bNkOJgHgqLy/XyJEj9eGHH5qOAgO4AQdAbA0dOlS1a9c+aHtBQYG2bt1qIBEQT6+//rq6dOmiRo0amY4CAygmAcTWFVdcoffee++gD7hDDjlEq1atMpQKiJfKykq98MILuvfee01HgSEUkwBiraioSNOmTTugoNy3bx/FJOCR8ePH6/zzz1fTpk1NR4EhFJMAYu+ss87SzJkzdfjhh0uSdu/erRUrVhhOBURfeXm5nn76aT3wwAOmo8AgikkAeeGMM87Q7Nmz1aRJEyWTSX311VemIwGR99prr+miiy7SMcccYzoKDOI5k4i0jRs3avfu3aZjIELWrFmjTp06SZJWrlxpOA0gNWnSRIcddpjpGFkrKytT27ZtNXPmWwRoKQAAH6dJREFUTB1xxBGm48AcHlqOaEskEmrTpo3pGAiJbdu2qWbNmqpXr57j+/bt26eNGzeqefPmASXLzvr169W0aVMVFPD0trhbvny5zj33XL3//vumo2Rt+PDh2rp1qwYPHmw6CszioeWItubNm2vhwoWmYyAkBg0apFatWumGG24wHaVaOnTooLFjx4a22IV35syZo5EjR5qOkbWtW7dqxIgRWrBggekoCAHWTAIAgKwMHjxY999/v+rXr286CkKAYhIAALi2dOlSLVy4UH379jUdBSFBMQkAAFz7+c9/rueff16HHEIJgX9jzSSAvJZIJBTV+xCrsicSCUk64DiqtqVvDypT+jarLLlmdLpm6eci2z6c3u/mtarrEdU/U5mMHz9eRx11lNq3b286CkKEYhJAXvPjQz+IYiK1j9SCsoqJoiY9Q9W2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\n",
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "generator = Generator()\n",
    "\n",
    "tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "LAMBDA = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def generator_loss(disc_generated_output, gen_output, target):\n",
    "    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)\n",
    "\n",
    "    # mean absolute error\n",
    "    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))\n",
    "\n",
    "    total_gen_loss = gan_loss + (LAMBDA * l1_loss)+tf.image.ssim(gen_output,target,max_val=1)\n",
    "\n",
    "    return total_gen_loss, gan_loss, l1_loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Discriminator():\n",
    "#    initializer = tf.random_normal_initializer(0., 0.02)\n",
    "\n",
    "    inp = tf.keras.layers.Input(shape=[480, 640, 3], name='input_image')\n",
    "    tar = tf.keras.layers.Input(shape=[480, 640, 3], name='target_image')\n",
    "\n",
    "    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 64, 64, channels*2)\n",
    "\n",
    "    down1 = downsample(32, 3, False)(x) # (bs, 32, 32, 32)\n",
    "    down2 = downsample(64, 3)(down1) # (bs, 16, 16, 64)\n",
    "    down3 = downsample(128, 3)(down2) # (bs, 8, 8, 128)\n",
    "\n",
    "    conv = tf.keras.layers.Conv2D(256, 3, strides=1,\n",
    "                                  padding='same',\n",
    "                                  use_bias=False)(down3) # (bs, 8, 8, 256)\n",
    "\n",
    "    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)\n",
    "\n",
    "    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)\n",
    "\n",
    "    last = tf.keras.layers.Conv2D(1, 3, strides=1)(leaky_relu) # (bs, 8, 8, 1)\n",
    "\n",
    "    return tf.keras.Model(inputs=[inp, tar], outputs=last)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "discriminator = Discriminator()\n",
    "#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def discriminator_loss(disc_real_output, disc_generated_output):\n",
    "    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)\n",
    "\n",
    "    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)\n",
    "\n",
    "    total_disc_loss = real_loss + generated_loss\n",
    "\n",
    "    return total_disc_loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)\n",
    "discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_images(model, test_input, tar):\n",
    "    prediction = model(test_input, training=True)\n",
    "    plt.figure(figsize=(16, 16))\n",
    "\n",
    "    display_list = [test_input[0], tar[0], prediction[0]]\n",
    "    title = ['Input Image', 'Ground Truth', 'Predicted Image']\n",
    "\n",
    "    for i in range(3):\n",
    "        plt.subplot(1, 3, i+1)\n",
    "        plt.title(title[i])\n",
    "    # getting the pixel values between [0, 1] to plot it.\n",
    "        plt.imshow(display_list[i] * 0.5 + 0.5)\n",
    "        plt.axis('off')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "cp_dir1='GS15U3'#设置路径\n",
    "cp_prefix1=os.path.join(cp_dir1,'ckpt')\n",
    "cp_dir2='GS25U3'#设置路径\n",
    "cp_prefix2=os.path.join(cp_dir2,'ckpt')\n",
    "cp_dir3='GS30U3'#设置路径\n",
    "cp_prefix3=os.path.join(cp_dir3,'ckpt')\n",
    "cp_dir4='GS50U3'#设置路径\n",
    "cp_prefix4=os.path.join(cp_dir4,'ckpt')\n",
    "cp_dir5='JY0.1U3'#设置路径\n",
    "cp_prefix5=os.path.join(cp_dir5,'ckpt')\n",
    "cp_dir6='JY0.05U3'#设置路径\n",
    "cp_prefix6=os.path.join(cp_dir6,'ckpt')\n",
    "checkpoint=tf.train.Checkpoint( gen=generator,dis=discriminator)#放入保存的东西"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "@tf.function\n",
    "def train_step(input_image, target, epoch):\n",
    "    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:\n",
    "        gen_output = generator(input_image, training=True)\n",
    "\n",
    "        disc_real_output = discriminator([input_image, target], training=True)\n",
    "        disc_generated_output = discriminator([input_image, gen_output], training=True)\n",
    "\n",
    "        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)\n",
    "        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)\n",
    "\n",
    "    generator_gradients = gen_tape.gradient(gen_total_loss,\n",
    "                                          generator.trainable_variables)\n",
    "    discriminator_gradients = disc_tape.gradient(disc_loss,\n",
    "                                               discriminator.trainable_variables)\n",
    "\n",
    "    generator_optimizer.apply_gradients(zip(generator_gradients,\n",
    "                                          generator.trainable_variables))\n",
    "    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,\n",
    "                                              discriminator.trainable_variables))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "#checkpoint.restore(tf.train.latest_checkpoint(cp_dir))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fit(train_ds, epochs, test_ds,cp_prefix):\n",
    "    for epoch in range(epochs+1):\n",
    "        for n, (input_image, target) in train_ds.enumerate():\n",
    "            \n",
    "            train_step(input_image, target, epoch)\n",
    "            print(\".\",end=\"\")\n",
    "        if epoch%50 == 0:\n",
    "            for example_input, example_target in test_ds.take(1):\n",
    "                generate_images(generator, example_input, example_target)\n",
    "                print(\"Epoch: \", epoch)\n",
    "        if (epoch+1)%50==0:\n",
    "                checkpoint.save(file_prefix=cp_prefix)#保存"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs=501"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "fit(dataset1,epochs,dataset1,cp_prefix1)\n",
    "generator.save(\"GS15.h5\")"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "fit(dataset2, epochs, dataset2,cp_prefix2)\n",
    "generator.save(\"GS25.h5\")"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "fit(dataset3, epochs, dataset3,cp_prefix3)\n",
    "generator.save(\"GS30.h5\")"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "fit(dataset4, epochs, dataset4,cp_prefix4)\n",
    "generator.save(\"GS50.h5\")"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "fit(dataset5, epochs, dataset5,cp_prefix5)\n",
    "generator.save(\"JY0.1.h5\")"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "checkpoint.restore(tf.train.latest_checkpoint(cp_dir6))\n",
    "fit(dataset6, 1, dataset6,cp_prefix6)\n",
    "generator.save(\"JY0.05.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n",
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n",
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n",
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n",
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n",
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n"
     ]
    }
   ],
   "source": [
    "new_model1= tf.keras.models.load_model(\"GS15.h5\")\n",
    "new_model2= tf.keras.models.load_model(\"GS25.h5\")\n",
    "new_model3= tf.keras.models.load_model(\"GS30.h5\")\n",
    "new_model4= tf.keras.models.load_model(\"GS50.h5\")\n",
    "new_model5= tf.keras.models.load_model(\"JY0.1.h5\")\n",
    "new_model6= tf.keras.models.load_model(\"JY0.05.h5\")\n",
    "\n",
    "imgs_path1 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise15_F\\GSnoise15_F_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path1.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "imgs_path2 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise25_F\\GSnoise25_F_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path2.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "imgs_path3 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise30_F\\GSnoise30_F_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path3.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "imgs_path4 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise50_F\\GSnoise50_F_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path4.sort(key=lambda x:int(x.split('\\\\')[8].split('.')[0]))\n",
    "imgs_path5 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.1_M_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path5.sort(key=lambda x:int(x.split('\\\\')[7].split('.')[0]))\n",
    "imgs_path6 = glob.glob(r'F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.05_M_U1U2HIGHout_low_outIMAGES\\*.jpg')\n",
    "imgs_path6.sort(key=lambda x:int(x.split('\\\\')[7].split('.')[0]))\n",
    "\n",
    "save_path1= r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise15_F\\GSnoise15_F_U1U2HIGHout_low_outIMAGESU3/\"\n",
    "save_path2= r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise25_F\\GSnoise25_F_U1U2HIGHout_low_outIMAGESU3/\"\n",
    "save_path3= r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise30_F\\GSnoise30_F_U1U2HIGHout_low_outIMAGESU3/\"\n",
    "save_path4= r\"F:\\Videodenoise\\dataset\\test\\GS\\GS_NOISE_F\\GSnoise50_F\\GSnoise50_F_U1U2HIGHout_low_outIMAGESU3/\"\n",
    "save_path5= r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.1_M_U1U2HIGHout_low_outIMAGESU3/\"\n",
    "save_path6= r\"F:\\Videodenoise\\dataset\\test\\JY\\JYnoise_M\\JYnoise0.05_M_U1U2HIGHout_low_outIMAGESU3/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'F:\\\\Videodenoise\\\\dataset\\\\test\\\\GS\\\\GS_NOISE_F\\\\GSnoise15_F\\\\GSnoise15_F_U1U2HIGHout_low_outIMAGES\\\\0.jpg'"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "imgs_path1[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "import imageio\n",
    "\n",
    "def U3(imgs_path,save_path):\n",
    "    for i in imgs_path:\n",
    "        img = load_and_preprocess1(i)\n",
    "        img = np.expand_dims(img,0)\n",
    "        pre_img = new_model.predict(img)\n",
    "        pre_img = np.squeeze(pre_img,0)\n",
    "        imageio.imwrite(save_path+str(i.split(\"\\\\\")[8].split(\".\")[0])+\".jpg\", pre_img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Lossy conversion from float32 to uint8. Range [-0.9998468160629272, 0.980692446231842]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9989244341850281, 0.9531088471412659]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9594354629516602, 0.9123501777648926]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9917576313018799, 0.9300995469093323]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9976449608802795, 0.6728066205978394]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9733619689941406, 0.9419622421264648]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9998363852500916, 0.9482609033584595]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9909498691558838, 0.991698682308197]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9975034594535828, 0.9965746998786926]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9406633377075195, 0.9930063486099243]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.999162495136261, 0.9959401488304138]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9240348935127258, 0.9783918261528015]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.992999792098999, 0.9945358633995056]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9988953471183777, 0.9985877275466919]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9883494973182678, 0.9997622966766357]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9715255498886108, 0.9903547763824463]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9408084750175476, 0.9302042722702026]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9133880734443665, 0.9999446868896484]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9170498847961426, 0.9997535943984985]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9627490639686584, 0.9915940761566162]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9767134189605713, 0.9940868616104126]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9990459680557251, 0.9943628311157227]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9984962344169617, 0.9125790596008301]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.8867452144622803, 0.9833100438117981]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9995587468147278, 0.9851948618888855]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9733636975288391, 0.9543321132659912]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9882228970527649, 0.9989220499992371]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9970613121986389, 0.9952353835105896]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9760646820068359, 0.9897468686103821]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9942725896835327, 0.8834861516952515]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9996532797813416, 0.9959669709205627]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9985910058021545, 0.9821215271949768]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9996663928031921, 0.9926518201828003]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9979946613311768, 0.9796176552772522]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9992985129356384, 0.9941955804824829]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9985907077789307, 0.9847663044929504]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9996477365493774, 0.992267906665802]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.997107207775116, 0.9943217039108276]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9915515780448914, 0.8931342363357544]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9991985559463501, 0.9998701214790344]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9987234473228455, 0.9873462915420532]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9623199105262756, 0.9654569029808044]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9927754402160645, 0.8795534372329712]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9640153050422668, 0.9683051109313965]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.977099597454071, 0.9522739052772522]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9964701533317566, 0.9902021288871765]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9967458844184875, 0.9984702467918396]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9407907724380493, 0.7821320295333862]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9977639317512512, 0.9621269106864929]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9908190965652466, 0.9893765449523926]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9557362794876099, 0.9972419142723083]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.999265193939209, 0.9929302334785461]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9981852173805237, 0.993416965007782]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9995187520980835, 0.996232807636261]. Convert image to uint8 prior to saving to suppress this warning.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Lossy conversion from float32 to uint8. Range [-0.999847412109375, 0.9205171465873718]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.967423141002655, 0.9533657431602478]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.8471001982688904, 0.6837988495826721]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9954800605773926, 0.9970455169677734]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.898140013217926, 0.9858355522155762]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9955531358718872, 0.9998327493667603]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9215213656425476, 0.999849796295166]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9892681837081909, 0.9953428506851196]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.906532883644104, 0.9997437000274658]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.8646135926246643, 0.935052752494812]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9430835843086243, 0.9939219355583191]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9565812945365906, 0.9975431561470032]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9997989535331726, 0.9955835342407227]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9643035531044006, 0.9956589937210083]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9784207940101624, 0.9570395350456238]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9270171523094177, 0.7391334772109985]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9883864521980286, 0.9160683155059814]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9249210357666016, 0.696140468120575]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9911152124404907, 0.9959474802017212]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9997526407241821, 0.9880586266517639]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.998635470867157, 0.970463752746582]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9707253575325012, 0.9965166449546814]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9995238780975342, 0.9976013898849487]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9487990140914917, 0.9133930802345276]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9987200498580933, 0.9999800324440002]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9686053991317749, 0.958091676235199]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9783722758293152, 0.9510224461555481]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9849305748939514, 0.9852069020271301]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9998345971107483, 0.9653470516204834]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.937729001045227, 0.978633463382721]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9611217379570007, 0.9765394330024719]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9981361031532288, 0.9818241596221924]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9848744869232178, 0.9302286505699158]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9987201690673828, 0.9999800324440002]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9686053991317749, 0.9580916166305542]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9783722758293152, 0.9510225057601929]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9849305748939514, 0.9852069020271301]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9998345971107483, 0.9653470516204834]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.937729001045227, 0.978633463382721]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9611217379570007, 0.9765394330024719]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9981359839439392, 0.9818241596221924]. Convert image to uint8 prior to saving to suppress this warning.\n",
      "Lossy conversion from float32 to uint8. Range [-0.9848744869232178, 0.9302286505699158]. Convert image to uint8 prior to saving to suppress this warning.\n"
     ]
    }
   ],
   "source": [
    "U3(imgs_path1,save_path1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "version": "3.7.10"
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
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
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
 "nbformat_minor": 4
}
