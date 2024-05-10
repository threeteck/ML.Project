from time import sleep

from tqdm import tqdm
import pandas as pd
import numpy as np
import requests

df = pd.read_csv('data/malicious_phish_fixed.csv')
print('Data Len:', len(df))
print()

# Extract html data using asyncio
import asyncio
import aiohttp
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import time

async def fetch_html(url, label, session):
    try:
        async with session.get(url) as response:
            if response.status == 200 and 'html' in response.headers['Content-Type']:
                text = await response.text()
                return (url, label, text, '')
    except Exception as e:
        return (url, label, '', str(e))

    return (url, label, '', 'FATAL')

async def fetch_all_html(df):
    connector = aiohttp.TCPConnector(limit=100)
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='Creating tasks'):
            url = row['url']
            label = row['type']
            tasks.append(fetch_html(url, label, session))

        return await tqdm_asyncio.gather(*tasks)

def run_once(i, slice):
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    html_data = asyncio.run(fetch_all_html(slice))

    out_df_unfiltered = pd.DataFrame(html_data, columns=['url', 'type', 'html', 'error'])
    out_df_unfiltered.to_parquet(f'./data/data_parts/html_data_unfiltered_{i}.parquet')

    print(f'Num filtered: {(out_df_unfiltered["html"].str.len() >= 5000).sum()}')
    print(f'Num exceptions: {(out_df_unfiltered["error"].str.len() > 0).sum()}')

idx = 370
slice_len = 1000
data_len = len(df)
total_idx = data_len / slice_len
while slice_len * idx < data_len:
    time_start = time.time()
    print(f'Slice {idx}/{int(total_idx)}')
    slice = df[idx * slice_len:(idx+1) * slice_len]
    run_once(idx, slice)
    idx += 1
    sleep(2)
    time_end = time.time()
    exec_time = time_end - time_start
    print(f'\nExec time: {exec_time:.1f}s (ETC {exec_time * (total_idx - idx) / 60:.1f}m)' )
    print()