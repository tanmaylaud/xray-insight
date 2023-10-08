from typing import Iterable
import json
from bs4 import BeautifulSoup
from pydantic.networks import AnyHttpUrl, HttpUrl
import logging
from crawlers import RadiopaediaCrawler, Case, CaseFindings

logger = logging.getLogger('radiopaedia.org')


class RadiopaediaOrgCrawler(RadiopaediaCrawler):

    @staticmethod
    def is_compatible(url: AnyHttpUrl) -> bool:
        return 'https://radiopaedia.org/cases/system/' in url

    def _seed_stack(self, seed):
        self._discovery_queue.add(seed)

    def _process_list_page(self, bs: BeautifulSoup, current_url: AnyHttpUrl) -> Iterable[Case]:
        res = bs.find_all('a', {'class': 'search-result search-result-case'})

        try:
            next_page_url = f"https://radiopaedia.org{bs.find_all('a', {'class': 'next_page'})[0]['href']}"
            logger.info(f"Found next page URL {next_page_url}")
            self._discover_url(next_page_url)
        except Exception:
            logger.warning(f"No next page URL found on page with URL {current_url}")

        for element in res:
            url = element['href']
            bs = self._get_html(f"https://radiopaedia.org{url}")
            if not bs:
                continue
            yield self._process_case_page(bs=bs, current_url=f"https://radiopaedia.org{url}")

    def _process_case_page(self, bs: BeautifulSoup, current_url: AnyHttpUrl):
        title = bs.find_all('h1', {'class': 'header-title'})[0].text
        try:
            certainity_level = bs.find_all('span', {'class': 'diagnostic-certainty-title'})[0].text
        except Exception:
            logger.warning(f"No certainity level found in URL {current_url}")
            certainity_level = "n.a"

        finding_lines = []
        try:
            finding_lines_div = bs.find_all('div', {'class': 'sub-section study-findings body'})[0]
        except Exception:
            logger.warning(f"No findings found in URL {current_url}")

        for line in finding_lines_div.find_all('p'):
            try:
                finding_lines.append(CaseFindings(text=line.text))
            except Exception:
                logger.warning(f"No findings found in URL {current_url}")
                continue

        img_viewer_div = bs.find_all('div', {'class': 'InlineStudyViewer'})[0]
        img_viewer_dict = json.loads(img_viewer_div.text)
        series = img_viewer_dict["study"]["series"]
        try:
            img_url = str(f"https://prod-images-static.radiopaedia.org/images/{series[0]['id']}/{series[0]['encodings']['thumbnailed_files'][0]['original']}")
        except Exception:
            img_url = "invalid"

        case = Case(title=title, certainity_level=certainity_level, finding_lines=finding_lines, image_download_url=img_url)
        logger.debug(f"Successfully created case for {title} with URL {img_url}")
        return case

    def _discover_url(self, url: HttpUrl):
        if not (url in self._processed_urls or url in self._failed_urls):
            self._discovery_queue.add(url)
        #else:
        #    self._request_duplication_count += 1