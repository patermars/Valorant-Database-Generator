import asyncio
import json
import re
from typing import List, Dict, Optional
from playwright.async_api import async_playwright
import pandas as pd

class VLRScraper:
    def __init__(self):
        self.base_url = "https://www.vlr.gg"
        self.data = []

    async def init_browser(self, playwright):
        self.browser = await playwright.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
        self.context = await self.browser.new_context(viewport={'width': 1920, 'height': 1080},
                                                     user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        self.page = await self.context.new_page()

    async def get_event_matches(self, event_url: str) -> List[str]:
        print(f"Fetching matches from: {event_url}")
        await self.page.goto(event_url, wait_until='domcontentloaded')
        await self.page.wait_for_timeout(2000)
        match_links = await self.page.eval_on_selector_all('a.wf-module-item.match-item', 'elements => elements.map(el => el.href)')
        return [link for link in match_links if link]

    async def scrape_match(self, match_url: str) -> Optional[Dict]:
        try:
            print(f"Scraping: {match_url}")
            await self.page.goto(match_url, wait_until='networkidle', timeout=60000)
            await self.page.wait_for_selector('.vm-stats-game[data-game-id="all"]', timeout=30000)
            await self.page.wait_for_selector('.match-header-link-name', timeout=10000)
            await self.page.wait_for_timeout(3000)

            try:
                for btn in await self.page.locator('.js-spoiler').all():
                    await btn.click()
                await self.page.wait_for_timeout(500)
            except:
                pass

            match_data = {
                'url': match_url,
                'match_id': match_url.split('/')[-2] if '/' in match_url else None,
            }

            match_data.update(await self._get_event_info())
            match_data.update(await self._get_teams_and_scores())
            match_data.update(await self._get_match_metadata())
            match_data.update(await self._get_betting_odds())
            match_data['maps'] = await self._get_map_data()
            match_data['player_stats'] = await self._get_player_stats()
            match_data['team_history'] = await self._get_team_history()
            match_data['h2h'] = await self._get_head_to_head()

            return match_data
        except:
            return None

    async def _get_event_info(self) -> Dict:
        try:
            event_name = await self.page.locator('.match-header-event .text-of').first.text_content()
            event_series = None
            try:
                event_series = await self.page.locator('.match-header-event-series').text_content()
            except:
                pass
            patch = None
            try:
                patch_text = await self.page.locator('div.match-header-note:has-text("Patch")').text_content()
                patch_match = re.search(r'Patch ([\d.]+)', patch_text)
                patch = patch_match.group(1) if patch_match else None
            except:
                pass
            return {
                'event_name': event_name.strip() if event_name else None,
                'event_series': event_series.strip() if event_series else None,
                'patch': patch
            }
        except:
            return {'event_name': None, 'event_series': None, 'patch': None}

    async def _get_teams_and_scores(self) -> Dict:
        try:
            team_elements = await self.page.locator('.match-header-link-name .wf-title-med').all_text_contents()
            team1 = team_elements[0].strip() if len(team_elements) > 0 else None
            team2 = team_elements[1].strip() if len(team_elements) > 1 else None
            score_text = await self.page.locator('.match-header-vs-score .js-spoiler').first.text_content()

            team1_score, team2_score, winner = None, None, None
            if score_text and ':' in score_text.strip():
                parts = score_text.strip().split(':')
                if len(parts) == 2:
                    team1_score = int(parts[0].strip()) if parts[0].strip().isdigit() else None
                    team2_score = int(parts[1].strip()) if parts[1].strip().isdigit() else None
            if team1_score is not None and team2_score is not None:
                winner = 1 if team1_score > team2_score else 2

            return {
                'team1': team1,
                'team2': team2,
                'team1_score': team1_score,
                'team2_score': team2_score,
                'winner': winner,
                'format': await self._get_match_format()
            }
        except:
            return {}

    async def _get_player_stats(self) -> Dict:
        stats = {'team1': [], 'team2': []}
        try:
            all_view = await self.page.locator('.vm-stats-game[data-game-id="all"]').first
            tables = await all_view.locator('table.wf-table-inset.mod-overview').all()
            for team_idx, table in enumerate(tables[:2]):
                rows = await table.locator('tbody tr').all()
                for row in rows:
                    player = {}
                    try:
                        name = await row.locator('.mod-player a .text-of').first.text_content()
                        player['name'] = name.strip() if name else None
                        team = await row.locator('.mod-player .ge-text-light').first.text_content()
                        player['team'] = team.strip() if team else None
                        flag = await row.locator('.flag').first.get_attribute('class')
                        if flag: player['country'] = flag.split('mod-')[-1].strip()
                    except:
                        continue
                    try:
                        agents = [await img.get_attribute('alt') for img in await row.locator('.mod-agents img').all()]
                        player['agents'] = [a for a in agents if a]
                    except:
                        player['agents'] = None
                    try:
                        player['rating'] = float((await row.locator('td:nth-child(3) .side.mod-both').text_content()).strip())
                        player['acs'] = int((await row.locator('td:nth-child(4) .side.mod-both').text_content()).strip())
                        player['kills'] = int((await row.locator('.mod-vlr-kills .side.mod-both').text_content()).strip())
                        player['deaths'] = int((await row.locator('.mod-vlr-deaths .side.mod-both').text_content()).strip())
                        player['assists'] = int((await row.locator('.mod-vlr-assists .side.mod-both').text_content()).strip())
                        player['kd_diff'] = (await row.locator('.mod-kd-diff .side.mod-both').text_content()).strip()
                        player['kast'] = (await row.locator('td:nth-child(9) .side.mod-both').text_content()).strip()
                        player['adr'] = int((await row.locator('td:nth-child(10) .side.mod-both').text_content()).strip())
                        player['hs_percent'] = (await row.locator('td:nth-child(11) .side.mod-both').text_content()).strip()
                        player['first_kills'] = int((await row.locator('.mod-fb .side.mod-both').text_content()).strip())
                        player['first_deaths'] = int((await row.locator('.mod-fd .side.mod-both').text_content()).strip())
                        player['fk_diff'] = (await row.locator('.mod-fk-diff .side.mod-both').text_content()).strip()
                    except:
                        pass
                    stats['team1' if team_idx == 0 else 'team2'].append(player)
            return stats
        except:
            return stats

    async def _get_map_data(self) -> List[Dict]:
        maps = []
        try:
            for map_elem in await self.page.locator('.vm-stats-game[data-game-id]').all():
                game_id = await map_elem.get_attribute('data-game-id')
                if not game_id or game_id == 'all':
                    continue
                map_data = {'game_id': game_id}
                try:
                    header = await map_elem.locator('.vm-stats-game-header').first
                    map_name_elem = await header.locator('.map div').first.text_content()
                    if map_name_elem:
                        map_data['map_name'] = map_name_elem.strip().split('\n')[0].strip()
                    try:
                        pick_elem = await header.locator('.picked').first.text_content()
                        map_data['pick_info'] = pick_elem.strip() if pick_elem else None
                    except:
                        map_data['pick_info'] = None
                    duration = await header.locator('.map-duration').text_content()
                    map_data['duration'] = duration.strip() if duration else None
                    scores = await header.locator('.score').all()
                    if len(scores) >= 2:
                        s1, s2 = await scores[0].text_content(), await scores[1].text_content()
                        map_data['team1_rounds'] = int(s1.strip()) if s1 and s1.strip().isdigit() else None
                        map_data['team2_rounds'] = int(s2.strip()) if s2 and s2.strip().isdigit() else None
                    for idx, score_elem in enumerate(await header.locator('.score').all()):
                        classes = await score_elem.get_attribute('class')
                        if 'mod-win' in classes:
                            map_data['map_winner'] = idx + 1
                            break
                except:
                    pass
                maps.append(map_data)
            return maps
        except:
            return []

    async def _get_match_format(self) -> str:
        try:
            format_text = await self.page.locator('.match-header-vs-note').nth(1).text_content()
            if format_text:
                cleaned = format_text.strip()
                bo_match = re.search(r'Bo(\d+)|Best of (\d+)', cleaned, re.IGNORECASE)
                if bo_match:
                    num = bo_match.group(1) or bo_match.group(2)
                    return f"Bo{num}"
                return cleaned
            return None
        except:
            return None

    async def _get_match_metadata(self) -> Dict:
        try:
            date_elem = await self.page.locator('.moment-tz-convert').first.get_attribute('data-utc-ts')
            picks_bans = await self.page.locator('.match-header-note').text_content()
            return {
                'date': date_elem if date_elem else None,
                'picks_bans': picks_bans.strip() if picks_bans else None
            }
        except:
            return {'date': None, 'picks_bans': None}

    async def _get_betting_odds(self) -> Dict:
        odds = {}
        try:
            for i, elem in enumerate(await self.page.locator('.match-bet-item').all()):
                text = await elem.text_content()
                odds_match = re.search(r'(\d+\.\d+)', text)
                if odds_match:
                    odds[f'odds_source_{i+1}'] = odds_match.group(1)
            return {'betting_odds': odds}
        except:
            return {'betting_odds': {}}

    async def _get_team_history(self) -> Dict:
        history = {'team1': [], 'team2': []}
        try:
            sections = await self.page.locator('.match-histories').all()
            for team_idx, section in enumerate(sections[:2]):
                matches = await section.locator('.match-histories-item').all()
                for match in matches[:5]:
                    result = await match.locator('.match-histories-item-result').text_content()
                    opponent = await match.locator('.match-histories-item-opponent-name').text_content()
                    date = await match.locator('.match-histories-item-date').text_content()
                    is_win = 'mod-win' in await match.get_attribute('class')
                    history['team1' if team_idx == 0 else 'team2'].append({
                        'result': result.strip() if result else None,
                        'opponent': opponent.strip() if opponent else None,
                        'date': date.strip() if date else None,
                        'win': is_win
                    })
            return history
        except:
            return history

    async def _get_head_to_head(self) -> List[Dict]:
        h2h = []
        try:
            for match in await self.page.locator('.match-h2h-matches .wf-module-item').all():
                event = await match.locator('.match-h2h-matches-event-name').text_content()
                score = await match.locator('.match-h2h-matches-score').text_content()
                date = await match.locator('.match-h2h-matches-date').text_content()
                h2h.append({'event': event.strip() if event else None,
                            'score': score.strip() if score else None,
                            'date': date.strip() if date else None})
            return h2h
        except:
            return []

    async def scrape_event(self, event_url: str, max_matches: int = None):
        match_urls = await self.get_event_matches(event_url)
        if max_matches:
            match_urls = match_urls[:max_matches]
        print(f"Found {len(match_urls)} matches to scrape")
        for url in match_urls:
            match_data = await self.scrape_match(url)
            if match_data:
                self.data.append(match_data)
                await asyncio.sleep(1)
        return self.data

    def save_to_json(self, filename: str = 'vlr_matches.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def to_dataframe(self) -> pd.DataFrame:
        flat_data = []
        for match in self.data:
            base = {
                'match_id': match.get('match_id'),
                'date': match.get('date'),
                'event_name': match.get('event_name'),
                'event_series': match.get('event_series'),
                'patch': match.get('patch'),
                'team1': match.get('team1'),
                'team2': match.get('team2'),
                'team1_score': match.get('team1_score'),
                'team2_score': match.get('team2_score'),
                'winner': match.get('winner'),
                'format': match.get('format'),
            }
            if 'player_stats' in match:
                for team in ['team1', 'team2']:
                    players = match['player_stats'].get(team, [])
                    if players:
                        avg_rating = sum(float(p.get('rating', 0)) for p in players if p.get('rating')) / len(players)
                        base[f'{team}_avg_rating'] = avg_rating
            if 'team_history' in match:
                for team in ['team1', 'team2']:
                    history = match['team_history'].get(team, [])
                    if history:
                        wins = sum(1 for m in history if m.get('win'))
                        base[f'{team}_recent_winrate'] = wins / len(history)
            flat_data.append(base)
        return pd.DataFrame(flat_data)

    async def close(self):
        await self.browser.close()


async def main():
    scraper = VLRScraper()
    async with async_playwright() as playwright:
        await scraper.init_browser(playwright)
        event_url = "https://www.vlr.gg/event/matches/2283/valorant-champions-2025"
        await scraper.scrape_event(event_url, max_matches=1)
        scraper.save_to_json('vlr_champions_2025.json')
        df = scraper.to_dataframe()
        df.to_csv('vlr_champions_2025.csv', index=False)
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())
