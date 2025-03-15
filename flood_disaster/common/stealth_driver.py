"""
Author: Joel
FilePath: nature/flood_disaster/common/stealth_driver.py
Date: 2025-03-05 12:10:44
LastEditTime: 2025-03-06 09:33:44
Description: 隐形的driver
"""
import random

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def get_stealth_driver():
    chrome_options = Options()

    # 禁用自动化特征
    chrome_options.add_argument("disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # 添加正常浏览器特征
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
    )
    width = random.randint(1024, 1400)
    height = random.randint(768, 1050)
    chrome_options.add_argument(f"window-size={width},{height}")

    # 禁用沙盒和GPU加速
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=chrome_options)

    # 执行JavaScript修改navigator.webdriver属性
    # driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")

    # 覆盖WebDriver属性
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
              get: () => undefined
            });
            """
    })
    return driver
