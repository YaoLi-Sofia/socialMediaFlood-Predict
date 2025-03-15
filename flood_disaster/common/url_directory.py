"""
Author: Joel
FilePath: nature/flood_disaster/common/url_directory.py
Date: 2025-03-05 14:02:18
LastEditTime: 2025-03-06 15:34:50
Description: url
"""


def get_url(arg):
    url = ""
    if arg == "douyin":
        url = "https://www.douyin.com/"
    elif arg == "weibo":
        url = "https://weibo.com/"
    elif arg == "facebook":
        url = "https://www.facebook.com/"
    elif arg == "twitter":
        url = "https://twitter.com/"
    elif arg == "xiaohongshu":
        url = "https://www.xiaohongshu.com/"
    elif arg == "bilibili":
        url = "https://www.bilibili.com/"
    elif arg == "search_bilibili":
        url = "https://search.bilibili.com/"

    return url
