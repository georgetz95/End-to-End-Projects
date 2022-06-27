def morph_transformations():
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    
    img = Image.open('./Data/PngItem_21450.png')
    img = np.array(img.resize((500,400)))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(img_gray, kernel)
    img_dilation = cv2.dilate(img_gray, kernel)
    img_opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    img_closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    img_gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
    img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    img_blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    
    fig, ax = plt.subplots(3,3,figsize=(10,10))
    imgs = [img, img_gray, img_erosion, img_dilation, img_opening, img_closing, img_gradient, img_tophat, img_blackhat]
    titles = ['Original', 'Grayscale', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Morphological Gradient', 'Top Hat', 'Black Hat']
    index = 0

    for x in range(3):
        for y in range(3):
            ax[x,y].imshow(imgs[index], cmap='gray')
            ax[x,y].set_title(titles[index], fontsize=20)
            ax[x,y].set_axis_off()
            index += 1
    
    fig.suptitle('Morphological Transformations of an Image', fontsize=25, fontweight='semibold')
    
    
morph_transformations()