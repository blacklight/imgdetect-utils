import argparse
import os
import shutil
import sys
import time

import cv2

boxes = []

def on_mouse(img):
    clicked = False

    def callback(event, x, y, flags, params):
        nonlocal clicked
        global boxes

        t = time.time()

        if event == cv2.EVENT_LBUTTONDOWN:
            print('Start Mouse Position: {},{}'.format(x,y))
            clicked = True
            boxes.append([(x, y)] * 2)

        if clicked:
            boxes[-1][1] = (x,y)

        if event == cv2.EVENT_LBUTTONUP:
            print('End Mouse Position: {},{}'.format(x,y))
            clicked = False

    return callback

def tag(img_dir, pos_dir, neg_dir, outfile):
    img_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
    images = [os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f)) and
            f.lower().split('.')[-1] in img_extensions]

    quit = False
    print('* Drag a rectangle on the image to identify the area(s) with a positive element')
    print('* Press ENTER to confirm an image')
    print('* Press c to clear the selected area(s)')
    print('* Press ESC/q to quit')

    for imgfile in images:
        img = cv2.imread(imgfile)
        orig_img = img.copy()
        boxes.clear()
        img_name = imgfile.split(os.sep)[-1]

        cv2.namedWindow(img_name)
        cv2.setMouseCallback(img_name, on_mouse(img), 0)

        while True:
            cv2.imshow(img_name, img)
            k = cv2.waitKey(1)

            img = orig_img.copy()

            for box in boxes:
                cv2.rectangle(img, box[0], box[1], color=(0, 0, 255), thickness=3)
                cv2.imshow(img_name, img)

            # Clear the tags on the image if c is pressed
            if k == ord('c'):
                boxes.clear()

            # Store and proceed to the next image upon ENTER
            if k == 13:
                is_positive = len(boxes) > 0
                dest_file = os.path.join(pos_dir if is_positive else neg_dir, img_name)
                shutil.move(imgfile, dest_file)
                print('Stored {} image {}'.format('positive' if is_positive else 'negative', dest_file))
                break

            # Quit upon 'q' or ESC
            if k == ord('q') or k == 27:
                quit = True
                break

        cv2.destroyAllWindows()

        if quit:
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', '-d', dest='dir', required=True,
                        help='Directory containing the images to be processed (supported formats: jpg, png, tiff, bmp)')
    parser.add_argument('--positive-dir', '-p', dest='pos_dir', required=True,
                        help='Directory that will store the images tagged as positive (those with at least one boundary box)')
    parser.add_argument('--negative-dir', '-n', dest='neg_dir', required=True,
                        help='Directory that will store the images tagged as negative (those with no boundary boxes)')
    parser.add_argument('--outfile', '-o', dest='outfile', required=True,
                        help='File that will contain the resulting trained classifier')

    opts, args = parser.parse_known_args(sys.argv[1:])
    tag(img_dir=opts.dir, pos_dir=opts.pos_dir, neg_dir=opts.neg_dir, outfile=opts.outfile)


if __name__ == '__main__':
    main()


# vim:sw=4:ts=4:et:
