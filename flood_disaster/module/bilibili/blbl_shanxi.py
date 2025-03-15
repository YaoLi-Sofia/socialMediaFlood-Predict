"""
Author: Joel
FilePath: nature/flood_disaster/module/bilibili/blbl_shanxi.py
Date: 2025-03-06 13:32:55
LastEditTime: 2025-03-06 16:07:31
Description: 
"""
import pandas as pd
import sys
from itertools import zip_longest
from pathlib import Path
import openpyxl

from bs4 import BeautifulSoup
from selenium.common import TimeoutException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

from common import stealth_driver, url_directory


def bilibili_get_flood():
    try:
        excel_path = current_dir / 'bili_shanxi.xlsx'
        if not excel_path.exists():
            pd.DataFrame().to_excel(excel_path)
        province_list = ['山西']
        keyword_list = ["洪灾", "洪水", "淹没", "淹水", "暴雨", "涨水", "水灾"]
        search_keyword = [province + keyword for province in province_list for keyword in keyword_list]

        for keyword in search_keyword:
            try:
                driver = stealth_driver.get_stealth_driver()
                url = f"{url_directory.get_url('search_bilibili')}all?keyword={keyword}"
                driver.get(url)
                driver.execute_script('localStorage.clear()')
                print(f"\n=== 开始检索：{keyword} ===")

                try:
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, '.bili-mini-close-icon'))
                    ).click()
                    print('登录弹窗已关闭')
                except TimeoutException:
                    print('未检测到登录弹窗')

                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.bili-video-card'))
                )

                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                video_tags = soup.select('.bili-video-card__info a')
                video_url_list = [item['href'] for item in video_tags if item.has_attr('href')]

                title_tags = soup.select('.bili-video-card__info .bili-video-card__info--tit')
                title_list = [item.get_text(strip=True) for item in title_tags]

                publisher_tags = soup.select('.bili-video-card__info--author')
                publish_date_tags = soup.select('.bili-video-card__info--date')
                publisher_list = [item.get_text(strip=True) for item in publisher_tags]
                publish_date_list = [item.get_text(strip=True).strip('·') for item in publish_date_tags]

                serial_number = [i + 1 for i in range(len(title_list))]

                filled_data = list(zip_longest(
                    title_list,
                    publisher_list,
                    publish_date_list,
                    video_url_list,
                    serial_number,
                    fillvalue='N/A'
                ))
                title_list, publisher_list, publish_date_list, video_url_list, serial_number = zip(*filled_data)

                data = {
                    "序号": serial_number,
                    "关键字": [keyword] * len(title_list),
                    "标题": title_list,
                    "发布人": publisher_list,
                    "发布日期": publish_date_list,
                    "视频链接": video_url_list,
                }

                # 在创建 DataFrame 前添加长度检查
                print(
                    f"长度检查 -> 序号: {len(serial_number)},"
                    f"标题: {len(title_list)}, "
                    f"发布人: {len(publisher_list)}, "
                    f"发布日期: {len(publish_date_list)}, "
                    f"视频链接: {len(video_url_list)}"
                )

                df = pd.DataFrame(data)

                if not excel_path.exists():
                    df.to_excel(excel_path, index=False, sheet_name='Sheet1')
                else:
                    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                        start_row = writer.sheets['Sheet1'].max_row
                        df.to_excel(writer, index=False, sheet_name='Sheet1', startrow=start_row, header=False)
                print(f"已保存 {len(df)} 条数据到 {excel_path}")

            except Exception as e:
                print(f"关键词：{keyword},发生错误：{str(e)}")
                driver.delete_all_cookies()
                driver.execute_script('localStorage.clear()')
                continue
            finally:
                driver.quit()
    except Exception as e:
        print(f"整体中发生错误：{str(e)}")


if __name__ == '__main__':
    bilibili_get_flood()
