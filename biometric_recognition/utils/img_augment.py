def cutout(img, gt_boxes, amount=0.5):
    '''
    ### Cutout ###
    img: image
    gt_boxes: format [[obj x1 y1 x2 y2],...]
    amount: num of masks / num of objects
    '''
    out = img.copy()
    ran_select = random.sample(gt_boxes, round(amount*len(gt_boxes)))

    for box in ran_select:
        x1 = int(box[1])
        y1 = int(box[2])
        x2 = int(box[3])
        y2 = int(box[4])
        mask_w = int((x2 - x1)*0.5)
        mask_h = int((y2 - y1)*0.5)
        mask_x1 = random.randint(x1, x2 - mask_w)
        mask_y1 = random.randint(y1, y2 - mask_h)
        mask_x2 = mask_x1 + mask_w
        mask_y2 = mask_y1 + mask_h
        cv2.rectangle(out, (mask_x1, mask_y1), (mask_x2, mask_y2), (0, 0, 0), thickness=-1)
    return out


def colorjitter(img, cj_type="b"):
	'''
	### Different Color Jitter ###
	img: image
	cj_type: {b: brightness, s: saturation, c: constast}
	'''
	if cj_type == "b":
		# value = random.randint(-50, 50)
		value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)
		if value >= 0:
			lim = 255 - value
			v[v > lim] = 255
			v[v <= lim] += value
		else:
			lim = np.absolute(value)
			v[v < lim] = 0
			v[v >= lim] -= np.absolute(value)

		final_hsv = cv2.merge((h, s, v))
		img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
		return img

	elif cj_type == "s":
		# value = random.randint(-50, 50)
		value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)
		if value >= 0:
			lim = 255 - value
			s[s > lim] = 255
			s[s <= lim] += value
		else:
			lim = np.absolute(value)
			s[s < lim] = 0
			s[s >= lim] -= np.absolute(value)

		final_hsv = cv2.merge((h, s, v))
		img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
		return img

	elif cj_type == "c":
		brightness = 10
		contrast = random.randint(40, 100)
		dummy = np.int16(img)
		dummy = dummy * (contrast / 127 + 1) - contrast + brightness
		dummy = np.clip(dummy, 0, 255)
		img = np.uint8(dummy)
		return img


def noisy(img, noise_type="gauss"):
	'''
	### Adding Noise ###
	img: image
	cj_type: {gauss: gaussian, sp: salt & pepper}
	'''
	if noise_type == "gauss":
		image = img.copy()
		mean = 0
		st = 0.7
		gauss = np.random.normal(mean, st, image.shape)
		gauss = gauss.astype('uint8')
		image = cv2.add(image, gauss)
		return image

	elif noise_type == "sp":
		image = img.copy()
		prob = 0.05
		if len(image.shape) == 2:
			black = 0
			white = 255
		else:
			colorspace = image.shape[2]
			if colorspace == 3:  # RGB
				black = np.array([0, 0, 0], dtype='uint8')
				white = np.array([255, 255, 255], dtype='uint8')
			else:  # RGBA
				black = np.array([0, 0, 0, 255], dtype='uint8')
				white = np.array([255, 255, 255, 255], dtype='uint8')
		probs = np.random.random(image.shape[:2])
		image[probs < (prob / 2)] = black
		image[probs > 1 - (prob / 2)] = white
		return image


def filters(img, f_type="blur"):
	'''
	### Filtering ###
	img: image
	f_type: {blur: blur, gaussian: gaussian, median: median}
	'''
	if f_type == "blur":
		image = img.copy()
		fsize = 9
		return cv2.blur(image, (fsize, fsize))

	elif f_type == "gaussian":
		image = img.copy()
		fsize = 9
		return cv2.GaussianBlur(image, (fsize, fsize), 0)

	elif f_type == "median":
		image = img.copy()
		fsize = 9
		return cv2.medianBlur(image, fsize)