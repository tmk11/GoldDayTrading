"""Data-fetching layer for GoldDayTrading.

Each module in this package returns either a structured object
(pandas DataFrame, dataclass) or a markdown block ready for prompt
injection. All fetchers degrade gracefully on network failures so a
single dead source does not stop the pipeline.
"""
