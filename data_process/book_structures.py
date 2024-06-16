from __future__ import annotations
from dataclasses import asdict, dataclass, field
import os
import re
import timeit
from typing import Generator
import numpy as np
import fitz
from paddleocr import PPStructure

BBox = tuple[float, float, float, float]

table_engine = PPStructure(table=False, ocr=False, show_log=False)  # 用于检测表格


@dataclass
class Boundaries:
    """
    边框界
    可以用来去除页眉页尾、表格
    默认是去除页眉页尾
    """

    boundaries: BBox = (0, 70, 1024, 635)

    def __contains__(self, bbox: BBox) -> bool:
        return (
            self.boundaries[0] <= bbox[0] <= self.boundaries[2]
            and self.boundaries[1] <= bbox[1] <= self.boundaries[3]
            and self.boundaries[0] <= bbox[2] <= self.boundaries[2]
            and self.boundaries[1] <= bbox[3] <= self.boundaries[3]
        )


@dataclass
class Subsection:
    name: str = ""
    paragraphs: str = ""


@dataclass
class Section:
    name: str = ""
    subsections: list[Subsection] = field(default_factory=list)


@dataclass
class Line:
    """
    页面上的一行
    对应fitz中的有"lines"的Block
    """

    block: dict

    def __post_init__(self) -> None:
        self.font, self.text, self.font_size, self.location = self.get_info()

    def get_info(self) -> tuple[str, str, float, BBox]:
        return (
            self.get_font(),
            self.get_text(),
            self.get_font_size(),
            self.get_location(),
        )

    def get_text(self) -> str:
        text: str = ""
        for line in self.block["lines"]:
            for span in line["spans"]:
                text += span["text"]

        # 去除图表标题
        if re.match(r"图.-.*|表.-.*[^所示]", text):
            return ""
        else:
            return text

    def get_font(self) -> str:
        return self.block["lines"][0]["spans"][0]["font"]

    def get_font_size(self) -> float:
        return self.block["lines"][0]["spans"][0]["size"]

    def get_location(self) -> tuple[float, float, float, float]:
        return self.block["bbox"]

    def is_start_of_paragraph(self, next_line: Line) -> bool:
        """
        判断是否是段落的开头
        """
        return (
            3 * self.font_size
            > next_line.location[0] - self.location[0]
            > 1.5 * self.font_size
        )


@dataclass
class Page:
    page: fitz.fitz.Page
    boundaries: Boundaries = Boundaries()
    skip_table: bool = False  # 跳过表格,默认不跳过

    def __post_init__(self) -> None:
        # 因为self.lines是生成器所以只能遍历一次
        self.lines: Generator[Line, None, None] = self.get_sorted_lines()

    def get_text(self, skip_table: bool | None = None) -> str:
        if skip_table is None:
            skip_table = self.skip_table
        if not skip_table:
            return self.page.get_text()
        else:
            lines: Generator[Line, None, None] = self.lines
            return "\n".join(line.text for line in lines)

    def get_pixmap(self) -> fitz.fitz.Pixmap:
        return self.page.get_pixmap()

    def get_sorted_lines(self) -> Generator[Line, None, None]:
        #  返回筛选过的行的迭代器
        return (
            Line(block)
            for block in sorted(
                self.page.get_text("dict")["blocks"], key=lambda b: b["bbox"][1]
            )
            if "lines" in block
            and (
                block["bbox"] in self.boundaries
                and (
                    not self.skip_table
                    or all(
                        block["bbox"] not in table_bbox
                        for table_bbox in self.detect_tables()
                    )
                )
            )
        )

    @staticmethod
    def expand_bbox(
        bbox: list[float], expansion: list[float] = [-1, -1, 1, 1]
    ) -> Boundaries:
        """
        考虑到fitz的get_pixmap()函数会将边界框四舍五入为int(原float64),
        所以需要扩大结果的边界框bbox以求一一对应回原pdf。

        :param bbox: 要扩大的边界框，格式为 [x_min, y_min, x_max, y_max]
        :param expansion: 扩大的量，格式为 [left, top, right, bottom]
        :return: 扩大后的边界框，格式为 [x_min, y_min, x_max, y_max]
        """
        return Boundaries(
            (
                bbox[0] - expansion[0],
                bbox[1] - expansion[1],
                bbox[2] + expansion[2],
                bbox[3] + expansion[3],
            )
        )

    def detect_tables(self) -> list[Boundaries]:
        pixmap: fitz.fitz.Pixmap = self.get_pixmap()
        image: np.ndarray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        detect_results: list[dict] = table_engine(image)
        return [
            self.expand_bbox(detect_result["bbox"])
            for detect_result in detect_results
            if detect_result["type"] == "table"
        ]


@dataclass
class Book:
    path: str
    sections: list[Section] = field(default_factory=list)
    subsection_font: str = "FZZZHUNHK--GBK1-0"
    section_font: str = "FZPSZHJW--GB1-0"
    current_page_index: int = 0
    skip_page: bool = False  # 跳过单元开头和单元结尾
    skip_table: bool = False  # 跳过表格

    def __post_init__(self) -> None:
        self.name: str = os.path.splitext(os.path.basename(self.path))[0]

    def __enter__(self) -> Book:
        self.pages = fitz.Document(self.path)
        return self

    def __exit__(self, *args) -> None:
        if self.pages:
            self.pages.close()

    def __iter__(self) -> Book:
        return self

    def __next__(self) -> Page:
        if self.current_page_index < len(self.pages):
            page = Page(self.pages[self.current_page_index], skip_table=self.skip_table)
            self.current_page_index += 1
            return page
        else:
            raise StopIteration

    def __getitem__(self, index: int) -> Page:
        return Page(self.pages[index], skip_table=self.skip_table)

    def __len__(self) -> int:
        return len(self.pages)

    @staticmethod
    def book_dict_factory(data) -> dict[str, str | list[Section | Subsection]]:
        return {
            k: v
            for k, v in data
            if k in ["name", "sections", "subsections", "paragraphs"]
        }

    def to_dict(self) -> dict:
        return asdict(self, dict_factory=Book.book_dict_factory)


if __name__ == "__main__":

    def example_table_detect(
        book_path: str = "./resources/数据与编程正文_三样.pdf", page_num: int = 10
    ) -> None:
        """
        预览检测表格效果
        """
        from PIL import Image
        from paddleocr import draw_structure_result

        with Book(book_path, skip_table=False) as book:
            page: Page = book[page_num]  # 指定要测试的页面
            print("忽略表格前：")
            print(page.get_text(), "\n")

            pixmap = page.get_pixmap()
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            result: list[dict] = table_engine(image)

            # 画图标注
            font_path = "./simfang.ttf"  # PaddleOCR下提供字体包
            im_show = draw_structure_result(image, result, font_path=font_path)
            im_show = Image.fromarray(im_show)
            im_show.save("page_detect_result.jpg")

            page.skip_table = True
            print("忽略表格后：")
            print(page.get_text())

    print(
        timeit.timeit(
            "example_table_detect()",
            setup="from __main__ import example_table_detect",
            number=1,
        ),
        "秒",
    )
