from abc import ABC, abstractmethod
from pydantic import AnyHttpUrl, BaseModel
from typing import Iterable, List
import time
import requests
from bs4 import BeautifulSoup
import logging
from crawlers.log import except2str

logger = logging.getLogger('RadiopaediaCrawler')


class CaseFindings(BaseModel):
    text: str


class Case(BaseModel):
    """
    This is the representation of a single case.
    """
    title: str
    certainity_level: str
    finding_lines: List[CaseFindings] = None
    image_download_url: str

    @property
    def findings(self):
        return ' '.join([v.text for v in self.finding_lines])


class RadiopaediaCrawler(ABC):

    def __init__(self, grace_period: float = 1.0):
        self._discovery_queue = set()
        self._processed_urls = set()
        self._failed_urls = set()
        self._grace_period = grace_period
        #self._request_count = 0
        #self._err_request_count = 0
        #self._err_extraction_count = 0
        #self._request_duplication_count = 0
        #self._processed = []
        #self._try_duplicate = []

    @staticmethod
    @abstractmethod
    def is_compatible(url: AnyHttpUrl) -> bool:
        raise NotImplementedError('This method should return if the given URL is understood by this crawler.')

    @abstractmethod
    def _seed_stack(self, seed: AnyHttpUrl):
        raise NotImplementedError('This method should use the seed URL to prime the crawl stack.')

    @abstractmethod
    def _process_list_page(self, bs: BeautifulSoup, current_url: AnyHttpUrl) -> Iterable[Case]:
        raise NotImplementedError('This method should use the seed URL to prime the crawl stack.')

    @abstractmethod
    def _discover_url(self, url: AnyHttpUrl):
        raise NotImplementedError('This method should append new unseen URLs to the URL stack.')

    def run(self, seed: AnyHttpUrl):
        self._seed_stack(seed)
        while len(self._discovery_queue) > 0:
            logger.info(f'Current number of URLs in the discovery queue: {len(self._discovery_queue)}, '
                        f'processed {len(self._processed_urls)} URLs and failed for {len(self._failed_urls)} URLs!')
            current_url = self._discovery_queue.pop()
            try:
                bs = self._get_html(current_url)
                if bs:
                    yield from self._process_list_page(bs, current_url)
                    self._processed_urls.add(current_url)
                else:
                    self._failed_urls.add(current_url)
                    #self._err_extraction_count += 1
            except Exception as err:
                self._failed_urls.add(current_url)
                logger.critical(f'Unforeseen error occurred: {except2str(err, logger)}')

        #print('Requests: ', self._request_count)
        #print('Failed request: ', self._err_request_count)
        #print('Failed extraction: ', self._err_extraction_count)
        #print('Double request: ', len(self._try_duplicate))
        #print('Double request list: ', self._try_duplicate)

    def _get_html(self, url: AnyHttpUrl) -> BeautifulSoup:
        time.sleep(self._grace_period)
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.debug(f'Successfully loaded: {url}')
            #self._request_count += 1
            return BeautifulSoup(response.text, 'lxml')
        except requests.HTTPError as http_err:
            logger.warning(f'Failed to load {url} due to {except2str(http_err)}')
            if http_err.response.status_code == 429:
                logger.warn("Too many requests, backing up for some time...")
                time.sleep(10)
            #self._err_request_count += 1
