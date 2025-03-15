"""
Author: Joel
FilePath: nature/flood_disaster/module/shanxi/dy_shanxi.py
Date: 2025-03-04 12:47:34
LastEditTime: 2025-03-06 10:58:26
Description: 抖音获取山西洪涝数据
"""

import sys
from itertools import zip_longest
from pathlib import Path
import random

import httpx
import os

import numpy
import openpyxl
import pandas as pd
from bs4 import BeautifulSoup
from openpyxl import Workbook
from openpyxl.reader.excel import load_workbook
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

current_dir = Path(__file__).resolve().parent  # 当前文件所在目录 (module/shanxi)
project_foot = current_dir.parent.parent  # 上两级目录 (flood_disaster)
sys.path.append(str(project_foot))  # 将根目录加入系统路径

from common import stealth_driver, url_directory


def douyin_get_shanxi_flood():
    result_dir = current_dir / 'results'
    excel_path = result_dir / 'shanxi.xlsx'

    # 使用改进后的driver初始化
    driver = stealth_driver.get_stealth_driver()

    try:
        # url = "https://www.douyin.com/search/%E5%B1%B1%E8%A5%BF%E6%B4%AA%E7%81%BE?aid=89d6649b-74b8-4ed8-ba52-23f1530654a9&type=general"
        province_list = ["山西"]
        city_list = []  # ["太原市", "大同市", "阳泉市", "长治市", "晋城市", "朔州市", "晋中市", "运城市", "忻州市", "临汾市","梁市"]
        district_list = []
        keyword_list = ["洪灾", "洪水", "淹没", "淹水", "暴雨", "涨水", "水灾"]
        search_keywords = [province + keyword for province in province_list for keyword in keyword_list]
        print(f"已获取关键字列表: {search_keywords}")
        for keyword in search_keywords:
            try:
                print(f"\n=== 开始处理关键词： {keyword} ===\n")

                url = url_directory.get_url('douyin')
                driver.get(url)
                driver.execute_script("localStorage.clear()")

                try:
                    # 初始化等待页面加载
                    login_close_btn = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, '.douyin-login__close.dy-account-close'))
                    )
                    WebDriverWait(driver, 2).until(
                        EC.element_to_be_clickable(login_close_btn)
                    )
                    login_close_btn.click()
                    print("登录弹窗已关闭")
                except TimeoutException:
                    print("未检测到登录页面")

                search_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.YEhxqQNi.jUqDCyab'))
                )
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable(search_input))

                search_input.clear()
                search_input.send_keys(keyword)

                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, '.YQK6ciuU'))
                ).click()
                print(f"已发起搜索：{keyword}")

                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, '.vqtFIVjM'))
                )

                # 记录以获取的视频数量
                last_count = 0
                current_count = 0
                retry_times = 0
                max_retry = 3  # 最大重试次数
                # 持续滚动直到没有新内容加载
                while retry_times < max_retry:
                    # 滚动前记录当前视频数量
                    current_count = len(driver.find_elements(By.CSS_SELECTOR, '.vqtFIVjM'))

                    # 先将鼠标移动到瀑布容器中
                    ActionChains(driver).move_to_element(driver.find_element(By.CSS_SELECTOR, '.irlccImO')).perform()

                    # # 鼠标移动到瀑布流容器
                    # ActionChains(driver).move_to_element(
                    #     driver.find_element(By.CSS_SELECTOR, '.st17zJnd')
                    # ).perform()
                    scroll_container = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, '.irlccImO'))
                    )
                    # 模拟人类滚动(带随机间隔)
                    scroll_distance = random.randint(800, 1500)
                    # driver.execute_script(f"window.scrollBy(200, {scroll_distance})")
                    driver.execute_script("arguments[0].scrollBy(0,arguments[1])", scroll_container, scroll_distance)

                    # container = driver.find_element(By.CSS_SELECTOR, '.irlccImO')
                    # pb_width = container.size['width'] - 1
                    # pb_height = container.size['height'] - 1
                    # 
                    # # 移动前鼠标的位置
                    # current_x = driver.execute_script("return window.screenX + (window.outerWidth / 2);")
                    # current_y = driver.execute_script("return window.screenY + (window.outerHeight / 2);")
                    # 
                    # # 生成安全偏移量
                    # safe_offset_x = random.randint(-50, 50)
                    # safe_offset_y = random.randint(-50, 50)
                    # # 计算安全位置
                    # new_x = max(0, min(pb_width, current_x + safe_offset_x))
                    # new_y = max(0, min(pb_height, current_y + safe_offset_y))

                    # 添加小幅随机移动鼠标
                    ActionChains(driver).move_to_element(
                        driver.find_element(By.CSS_SELECTOR, '.vqtFIVjM')).perform()
                    ActionChains(driver).move_by_offset(-5, 5).perform()

                    # 随机等待时间（2-5秒）
                    time.sleep(random.uniform(2.5, 5.5))

                    # 检查是否有新内容加载
                    new_count = len(driver.find_elements(By.CSS_SELECTOR, '.vqtFIVjM'))
                    if new_count == current_count:
                        retry_times += 1
                        print(f"未加载新内容，重试次数：{retry_times}/{max_retry}")
                    else:

                        retry_times = 0  # 重置计数器
                        print(f"已加载视频数：{new_count}")

                    # 最终等待所有元素加载完成
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.oyfanDG1 img'))
                    )

                # driver.implicitly_wait(15)

                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                title_items = soup.select('.vqtFIVjM')
                image_items = soup.select('.oyfanDG1 img')

                publisher = soup.select('.VikzymRj')
                publish_date = soup.select('.wTD2qIyI')

                image_url_list = [item['src'] for item in image_items if item.has_attr('src')]
                title_list = [item.get_text(strip=True) for item in title_items]
                publisher_list = [item.get_text(strip=True) for item in publisher]
                publish_date_list = [item.get_text(strip=True).replace('·', '').strip() for item in publish_date]
                serial_number = [i + 1 for i in range(len(title_list))]

                # 用 None 填充短数组
                filled_data = list(
                    zip_longest(image_url_list, title_list, publisher_list, publish_date_list, serial_number,
                                fillvalue=None))

                # 重新拆分为等长列表
                image_url_list, title_list, publisher_list, publish_date_list, serial_number = zip(
                    *filled_data)  # 元组转列表

                data = {
                    'serial_number': serial_number,
                    'keyword': keyword,
                    'title': title_list,
                    'publisher': publisher_list,
                    'publish_date': publish_date_list,
                    'image_url': image_url_list,
                }
                df = pd.DataFrame(data)
                if not excel_path.exists():
                    df.to_excel(excel_path, index=False, sheet_name='Sheet1')
                else:
                    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                        start_row = writer.sheets['Sheet1'].max_row
                        df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=start_row, header=False)
                print(f"已保存 {len(df)} 条数据到 {excel_path}")
            except Exception as e:
                print(f"关键词：{keyword}:发生错误，{str(e)}")
                # 发生错误时重置浏览器状态
                driver.delete_all_cookies()
                driver.execute_script("localStorage.clear();")
                continue  # 继续下一个关键词
    except Exception as e:
        print(f"整体错误，{str(e)}")
    finally:
        driver.quit()


if __name__ == '__main__':
    douyin_get_shanxi_flood()
