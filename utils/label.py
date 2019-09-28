import argparse
import os
import shutil
import sys

import cv2


def label(img_dir, scale_factor):
    img_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
    images = sorted([os.path.join(img_dir, f)
                     for f in os.listdir(img_dir)
                     if os.path.isfile(os.path.join(img_dir, f)) and
                     f.lower().split('.')[-1] in img_extensions])

    labels = [f for i, f in enumerate(sorted(os.listdir(img_dir)))
              if os.path.isdir(os.path.join(img_dir, f))]

    if not labels:
        raise RuntimeError('No subdirectories found. Please create subdirectories for ' +
                           'the labels you want to store (e.g. "negative", "positive")')

    for imgfile in images:
        img = cv2.imread(imgfile)
        img_name = os.path.basename(imgfile)

        if scale_factor != 1:
            size = (int(img.shape[0]*scale_factor), int(img.shape[1]*scale_factor))
            img = cv2.resize(img, size)

        print('[{}] Keys:'.format(os.path.basename(imgfile)))

        for i, l in enumerate(labels):
            print('\t({}): Tag image as "{}"'.format(i+1, l))

        print('\t(s): Skip this image')
        print('\t(d): Delete this image')
        print('\t(ESC/q): Quit the script')
        print('')

        cv2.namedWindow(img_name)
        cv2.imshow(img_name, img)
        k = cv2.waitKey()
        print('')

        if k == ord('c'):
            continue

        if ord('0') <= k <= ord('9'):
            label_index = int(chr(k))-1
            if label_index >= len(labels):
                print('Invalid label index "{}", skipping image'.format(label_index))

            shutil.move(imgfile, os.path.join(img_dir, labels[label_index]))

        if k == ord('d'):
            os.unlink(imgfile)

        # Quit upon 'q' or ESC
        if k == ord('q') or k == 27:
            break

        print('')
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', '-d', dest='dir', required=True,
            help='Directory that contains the images to be processed ' +
            '(supported formats: jpg, png, tiff, bmp)')
    parser.add_argument('--scale-factor', '-s', dest='scale', required=False, default=1,
            type=float, help='Scale factor to be applied to the images for visualization (default: 1)')

    opts, args = parser.parse_known_args(sys.argv[1:])
    label(img_dir=opts.dir, scale_factor=opts.scale)


if __name__ == '__main__':
    main()


# vim:sw=4:ts=4:et:
