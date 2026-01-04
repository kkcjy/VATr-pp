import os
import shutil

import cv2
import numpy as np

from data.dataset import CollectionTextDataset, TextDataset
from generate.util import stack_lines
from generate.writer import Writer


def generate_authors(arguments):
    # 加载数据集
    author_dataset = CollectionTextDataset(
        arguments.dataset, 
        'files', 
        TextDataset, 
        file_suffix=arguments.file_suffix, 
        num_examples=arguments.num_examples,
        collator_resolution=arguments.resolution, 
        validation=arguments.test_set
    )

    arguments.num_writers = author_dataset.num_writers

    # 初始化生成器
    style_generator = Writer(arguments.checkpoint, arguments, only_generator=True)

    # 读取文本内容
    if arguments.text.endswith(".txt"):
        with open(arguments.text, 'r') as text_file:
            input_lines = [line.rstrip() for line in text_file]
    else:
        input_lines = [arguments.text]

    # 创建输出目录
    output_directory = "saved_images/author_samples/"
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # 生成作者风格图像
    generated_images, author_identifiers, reference_styles = style_generator.generate_authors(
        input_lines, 
        author_dataset, 
        arguments.align, 
        arguments.at_once
    )

    # 保存生成的图像
    for batch_images, author_id, style_references in zip(generated_images, author_identifiers, reference_styles):
        author_output_dir = os.path.join(output_directory, str(author_id))
        os.makedirs(author_output_dir)

        # 保存每一行生成的图像
        for line_index, line_image in enumerate(batch_images):
            cv2.imwrite(
                os.path.join(author_output_dir, f"line_{line_index}.png"), 
                line_image
            )

        # 合并所有行并保存
        combined_image = stack_lines(batch_images)
        cv2.imwrite(
            os.path.join(author_output_dir, "combined.png"), 
            combined_image
        )

        # 保存参考风格图像（如果需要）
        if arguments.output_style:
            for style_index, style_image in enumerate(style_references):
                cv2.imwrite(
                    os.path.join(author_output_dir, f"style_reference_{style_index}.png"), 
                    style_image
                )