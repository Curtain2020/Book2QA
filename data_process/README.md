# 处理PDF书籍

本项目的作用是将格式为pdf的教科书电子版转换为结构化的json数据，式样如/resourses中pdf所示。

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [安装](#安装)
- [使用](#使用)
  - [将pdf转换为json](#将pdf转换为json)
  - [可视化表格检测的结果](#可视化表格检测的结果)
- [常见问题](#常见问题)
  - [找不到`paddlepaddle`指定版本](#找不到paddlepaddle指定版本)
  - [`paddlepaddle`提示找不到`libcuda.so`](#paddlepaddle提示找不到libcudaso)
- [限制](#限制)
- [后续计划](#后续计划)

<!-- /code_chunk_output -->

## 安装

项目环境为linux，windows是否兼容未测试。执行以下命令安装依赖：

```bash
conda env create -f environment.yml
```

## 使用

### 将pdf转换为json

将待转换的pdf文件放入`resources`文件夹，然后运行`book2json.py`，结果保存在`struct_books.json`中。

### 可视化表格检测的结果

修改`book_structures.py`中`if __name__ == "__main__"`下的代码并运行，可以显示pdf书籍某一页的表格检测结果，以及前后的文本提取差异，可视化结果保存在`page_dect_result.jpg`中。

## 常见问题

### 找不到`paddlepaddle`指定版本

安装环境时如果提示找不到`paddlepaddle`指定版本。可以使用官网的命令安装。(以下为Linux下命令)
`pip`安装：

```bash
python -m pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

`conda`安装：

```bash
conda install paddlepaddle-gpu==2.5.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

### `paddlepaddle`提示找不到`libcuda.so`

安装完`paddlepaddle`之后可以运行以下命令进行检查：

```bash
python -c "import paddle; paddle.utils.run_check()"
```

如果出现以下错误提示：

```python
The third-party dynamic library (libcuda.so) that Paddle depends on is not configured correctly. (error code is libcuda.so: cannot open shared object file: No such file or directory)
```

可以尝试使用以下命令找到libcuda.so所在：

```bash
sudo find /usr/ -name libcuda.so
```

找到后将该文件复制进你的虚拟环境下`your/path/to/anaconda3/envs/your_virtual_env_name/lib`文件夹中即可。

如果是提示找不到`libcudnn.so`，替换路径并运行下列命令：

```bash
echo "export LD_LIBRARY_PATH=~/anaconda3/envs/paddle/lib">>~/.bashrc
```

## 限制

- 目前仅针对这种样式的pdf采取基于规则的方法，适用性不强。
- 没有完全还原原来的换行情况，例如(1)xxx后没有换行。
- 大约有百分之几的图标标题没有去除。
- 少量图片会识别为text。

## 后续计划

- [ ] 优化表格检测性能(1.35s->293s)
  - [ ] 检查是否是因为频繁调用`table_engine`的问题，测试先检测出所有表格bbox而不是一页一检测的情况。
- [ ] 拓展到其他需要处理的数据类型上。
- [ ] 将json格式的结构化数据拼接为大模型需要的数据样式。
- [ ] 将数据转换为向量。
- [ ] 将向量存入向量数据库。
- [ ] 添加命令行运行的功能。
