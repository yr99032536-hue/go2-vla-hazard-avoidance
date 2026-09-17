from setuptools import find_packages, setup

package_name = "go2_active_slam"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="iy",
    maintainer_email="iy@example.com",
    description="Simulation-first deterministic active SLAM supervisor.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gap_arm_supervisor = go2_active_slam.supervisor_node:main",
            "tmaze_supervisor = go2_active_slam.tmaze_supervisor:main",
            "binary_tree_hazard_supervisor = go2_active_slam.binary_tree_hazard_supervisor:main",
            "maze_supervisor = go2_active_slam.maze_supervisor:main",
            "tsdf_fusion = go2_active_slam.tsdf_fusion_node:main",
            "nbv_teacher_collector = go2_active_slam.nbv_teacher_collector:main",
        ],
    },
)
