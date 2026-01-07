from setuptools import setup
from setuptools import find_packages

package_name = 'voice_llm_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='cletuzz@gmail.com',
    description='ROS2 package for voice LLM pipeline with STT, LLM, and TTS nodes',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stt_node = voice_llm_pipeline.stt_node:main',
            'llm_node = voice_llm_pipeline.llm_node:main',
            'tts_node = voice_llm_pipeline.tts_node:main',
        ],
    },
)

