from glob import glob

from setuptools import find_packages, setup


package_name = "s622_dual_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/models", glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dsl",
    maintainer_email="1197226282@qq.com",
    description="S622 dual-arm YOLO OBB and segmentation perception package.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_obb_detector = "
            "s622_dual_perception.nodes.yolo_detector_obb:main",
            "yolo_terminal_segmenter = "
            "s622_dual_perception.nodes.yolo_terminal_segmenter:main",
            "target_acquisition = "
            "s622_dual_perception.nodes.target_acquisition:main",
        ],
    },
)
