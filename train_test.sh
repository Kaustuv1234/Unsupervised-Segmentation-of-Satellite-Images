


# python train.py --dataset Vaihingen --no_patches 690 --modelName Model_BYOL
# python train.py --dataset Potsdam_RGB --no_patches 690 --modelName Model_BYOL
# python train.py --dataset Potsdam_RGBIR --no_patches 690 --modelName Model_BYOL

python train.py --dataset Vaihingen --no_patches 690 --modelName Model6
python train.py --dataset Potsdam_RGB --no_patches 690 --modelName Model6
python train.py --dataset Potsdam_RGBIR --no_patches 690 --modelName Model6


# python test.py --dataset Vaihingen --modelName Model_BYOL --classes 3
# python test.py --dataset Vaihingen --modelName Model_BYOL --classes 3 --superpixel_refine
# python test.py --dataset Potsdam_RGB --modelName Model_BYOL --classes 3
# python test.py --dataset Potsdam_RGB --modelName Model_BYOL --classes 3 --superpixel_refine
# python test.py --dataset Potsdam_RGBIR --modelName Model_BYOL --classes 3
# python test.py --dataset Potsdam_RGBIR --modelName Model_BYOL --classes 3 --superpixel_refine

# python test.py --dataset Vaihingen --modelName Model6 --classes 3
# python test.py --dataset Vaihingen --modelName Model6 --classes 3 --superpixel_refine
# python test.py --dataset Potsdam_RGB --modelName Model6 --classes 3
# python test.py --dataset Potsdam_RGB --modelName Model6 --classes 3 --superpixel_refine
# python test.py --dataset Potsdam_RGBIR --modelName Model6 --classes 3
# python test.py --dataset Potsdam_RGBIR --modelName Model6 --classes 3 --superpixel_refine

# python test.py --dataset Vaihingen --modelName Model_BYOL --classes 6
# python test.py --dataset Vaihingen --modelName Model_BYOL --classes 6 --superpixel_refine
# python test.py --dataset Potsdam_RGB --modelName Model_BYOL --classes 6
# python test.py --dataset Potsdam_RGB --modelName Model_BYOL --classes 6 --superpixel_refine
# python test.py --dataset Potsdam_RGBIR --modelName Model_BYOL --classes 6
# python test.py --dataset Potsdam_RGBIR --modelName Model_BYOL --classes 6 --superpixel_refine

# python test.py --dataset Vaihingen --modelName Model6 --classes 6
python test.py --dataset Vaihingen --modelName Model6 --classes 6 --superpixel_refine
# python test.py --dataset Potsdam_RGB --modelName Model6 --classes 6
python test.py --dataset Potsdam_RGB --modelName Model6 --classes 6 --superpixel_refine
# python test.py --dataset Potsdam_RGBIR --modelName Model6 --classes 6
python test.py --dataset Potsdam_RGBIR --modelName Model6 --classes 6 --superpixel_refine


