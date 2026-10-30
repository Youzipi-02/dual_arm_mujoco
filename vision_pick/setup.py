from glob import glob
from setuptools import find_packages, setup

setup(name='vision_pick', version='0.1.0', packages=find_packages(exclude=['test']),
      data_files=[('share/ament_index/resource_index/packages', ['resource/vision_pick']),
                  ('share/vision_pick', ['package.xml', 'README.md']),
                  ('share/vision_pick/config', glob('config/*.yaml')),
                  ('share/vision_pick/launch', glob('launch/*.launch.py'))],
      install_requires=['setuptools'], zip_safe=True, maintainer='dsl',
      maintainer_email='1197226282@qq.com', description='Vision-guided dual-arm picking with pymoveit2.',
      license='Apache-2.0', tests_require=['pytest'],
      entry_points={'console_scripts': ['vision_pick = vision_pick.node:main']})
