#!/usr/bin/env python3

import os
import fnmatch

BASE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
OUT = os.path.join(BASE, 'merged_python.txt')


def is_empty_init_py(filepath):
    """检查是否是空的__init__.py文件"""
    if os.path.basename(filepath) == '__init__.py':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 移除注释和空行后检查是否为空
                lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        lines.append(line)
                return len(lines) == 0
        except:
            return False
    return False


def collect_py_files():
    """递归收集 BASE 下所有非空的 *.py 文件，排除空__init__.py和脚本自身"""
    self_path = os.path.abspath(__file__)
    py_files = []

    for root, dirs, files in os.walk(BASE):
        for name in fnmatch.filter(files, '*.py'):
            filepath = os.path.join(root, name)

            # 排除脚本自身
            if os.path.abspath(filepath) == self_path:
                continue

            # 排除空的__init__.py
            if is_empty_init_py(filepath):
                print(f"跳过空文件: {os.path.relpath(filepath, BASE)}")
                continue

            py_files.append(filepath)

    return py_files


def generate_tree_structure(files):
    """生成漂亮的树状结构图"""
    tree_lines = []

    # 添加项目根目录
    base_name = os.path.basename(BASE) or "当前目录"
    tree_lines.append(f"{base_name}/")

    # 按目录分组文件
    dir_structure = {}
    for file in files:
        rel_path = os.path.relpath(file, BASE)
        dir_name = os.path.dirname(rel_path)
        file_name = os.path.basename(file)

        if dir_name == '':
            dir_name = '.'

        if dir_name not in dir_structure:
            dir_structure[dir_name] = []
        dir_structure[dir_name].append(file_name)

    # 按目录层级排序
    sorted_dirs = sorted(dir_structure.keys(), key=lambda x: (x.count(os.sep), x))

    # 生成树状图
    for i, dir_name in enumerate(sorted_dirs):
        is_last_dir = (i == len(sorted_dirs) - 1)

        if dir_name == '.':
            prefix = "├── " if not is_last_dir else "└── "
        else:
            prefix = "├── " if not is_last_dir else "└── "
            # 显示目录
            dir_prefix = "│   " if not is_last_dir else "    "
            tree_lines.append(f"{prefix}{dir_name}/")

        # 该目录下的文件
        files_in_dir = sorted(dir_structure[dir_name])
        dir_prefix = ("│   " if not is_last_dir else "    ") if dir_name != '.' else ""

        for j, file_name in enumerate(files_in_dir):
            is_last_file = (j == len(files_in_dir) - 1)
            file_prefix = "└── " if is_last_file else "├── "

            if dir_name == '.':
                tree_lines.append(f"{file_prefix}{file_name}")
            else:
                tree_lines.append(f"{dir_prefix}{file_prefix}{file_name}")

    return tree_lines


def main():
    """主函数"""
    print(f"正在扫描目录: {BASE}")

    # 收集所有.py文件
    py_files = collect_py_files()

    if not py_files:
        print("未找到任何.py文件！")
        return

    print(f"\n找到 {len(py_files)} 个Python文件:")

    # 生成树状结构
    tree_lines = generate_tree_structure(py_files)

    # 输出树状结构到控制台
    print("\n" + "=" * 60)
    print("项目结构树状图:")
    print("=" * 60)
    for line in tree_lines:
        print(line)

    # 写入输出文件
    with open(OUT, 'w', encoding='utf-8') as out:
        # 写入标题
        out.write("=" * 60 + "\n")
        out.write("Python项目源码合并文件\n")
        out.write("=" * 60 + "\n\n")

        # 写入树状结构
        out.write("项目结构树状图:\n")
        out.write("-" * 40 + "\n")
        for line in tree_lines:
            out.write(line + "\n")

        out.write("\n" + "=" * 60 + "\n\n")

        # 写入文件内容
        for idx, filepath in enumerate(sorted(py_files, key=lambda x: os.path.relpath(x, BASE))):
            rel_path = os.path.relpath(filepath, BASE)

            # 写入文件名
            out.write(f"文件: {rel_path}\n")
            out.write("-" * 40 + "\n")

            try:
                # 读取并写入文件内容
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    out.write(content)

                # 如果不是最后一个文件，添加分隔符
                if idx != len(py_files) - 1:
                    out.write("\n" + "=" * 60 + "\n\n")

            except Exception as e:
                out.write(f"读取文件时出错: {str(e)}\n")
                if idx != len(py_files) - 1:
                    out.write("\n" + "=" * 60 + "\n\n")

    print(f"\n" + "=" * 60)
    print(f"已成功汇总 {len(py_files)} 个Python文件 → {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()