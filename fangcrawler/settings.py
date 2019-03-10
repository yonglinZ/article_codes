ROBOTSTXT_OBEY = False

DOWNLOAD_DELAY = 1

DOWNLOADER_MIDDLEWARES = {
   'fang.middlewares.UserAgentDownloadMiddleware': 543,
}

ITEM_PIPELINES = {
   'fang.pipelines.FangPipeline': 300,
}