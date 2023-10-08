import argparse
import logging
from crawlers.log import init_logging
from crawlers.radiopaedia_org import RadiopaediaOrgCrawler

CRAWLERS = [
    RadiopaediaOrgCrawler
]

parser = argparse.ArgumentParser(description='Radiopaedia Scraper')
parser.add_argument('--seed', type=str,
                    default='https://radiopaedia.org/cases/system/chest?lang=us&modality=X-ray',
                    help='Seed URL the dedicated crawler expects to start at.')
parser.add_argument('--output', type=str, default='result.json',
                    help='Target file to store scrape results in')
parser.add_argument('--grace', type=float, default=1.0,
                    help='Time to sleep between each request')
parser.add_argument('--log', type=str, default='DEBUG',
                    choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                    help='Log Level')
args = parser.parse_args()

init_logging(args.log)
logger = logging.getLogger('main')

logger.debug('debug')
logger.info('info')
logger.warning('warn')
logger.error('error')
logger.fatal('fatal')

for _crawler in CRAWLERS:
    if _crawler.is_compatible(args.seed):
        logger.info(f'Determined {_crawler.__name__} as the fitting crawler for seed URL: {args.seed}')
        crawler = _crawler(grace_period=args.grace)
        with open(args.output, 'w') as f:
            for doc in crawler.run(args.seed):
                #f.write((doc.json() + '\n').encode('utf8'))
                f.write(doc.json() + '\n')
    else:
        logger.info(f'Crawler {_crawler.__name__} is incompatible with seed URL: {args.seed}')
