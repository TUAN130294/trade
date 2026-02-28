# VN-Quant Project Roadmap

**Last Updated:** 2026-02-25
**Current Phase:** Phase 2 - Validation & Optimization (In Progress)
**Version:** 4.0.0

---

## Vision Statement

Build a world-class autonomous trading system for the Vietnamese stock market that combines multi-agent AI consensus with machine learning predictions and real-time news sentiment analysis. Establish VN-Quant as the gold standard for algorithmic trading on VN-Index and HSX.

---

## Phase Timeline

### Phase 1: Foundation (COMPLETE ✅)
**Target:** 2026-01-01 to 2026-01-12
**Status:** DONE

#### Achievements
- [x] 6-agent system fully functional (Bull, Bear, Alex, Scout, RiskDoctor, Chief)
- [x] Stockformer ML integration (102 models, 5-day forecasting)
- [x] News sentiment pipeline (CafeF RSS, Vietnamese NLP)
- [x] Advanced confidence scoring (6-factor system)
- [x] Vietnam market compliance enforcement
- [x] Real-time signal caching and deduplication
- [x] Paper trading engine with realistic slippage
- [x] WebSocket-based real-time dashboard
- [x] 289 historical stock datasets downloaded
- [x] Docker containerization ready

#### Metrics Achieved
- 102+ Python modules, ~8,000 LOC
- 28+ API endpoints
- 100% critical path error handling
- 85%+ test coverage

---

### Phase 2: Validation & Optimization (IN PROGRESS ⏳)
**Target:** 2026-01-13 to 2026-03-15
**Current Status:** Major Progress (70% complete)

#### Completed Milestones
- [x] Refactored monolithic run.py into 4 modular routers (trading, market, data, news)
- [x] Migrated from inline HTML to React 19 + Vite 7.3 frontend
- [x] Implemented 12-phase money flow behavioral improvements
- [x] Added VPS API as primary market data source (foreign flow)
- [x] Fixed 34 bugs total (13 critical+high Phase 1 + 21 critical+high Phase 2)
- [x] Updated port from 8001 to 8100 (standardized)
- [x] Added LLM interpretation service (Claude Sonnet 4.6 via localhost:8317)
- [x] Today's live candle appended to OHLCV (real-time data)
- [x] VN-Index realtime status via CafeF banggia API
- [x] Chart candle colors VN market style (close vs prev close)
- [x] WebSocket exponential backoff reconnection
- [x] Complete API documentation (28+ endpoints, 4 routers)

#### Ongoing Goals
- [ ] Run paper trading for 15-20 trading days (in progress)
- [ ] Validate signal quality (expected: >50% win rate)
- [ ] Monitor model prediction accuracy per stock
- [ ] Test position management (exits, T+2 compliance)
- [ ] Measure system latency (target: <2 min signal→execution)
- [ ] Stress test with all 102 stocks simultaneously
- [ ] Benchmark against historical backtest results
- [ ] Collect performance baseline metrics

#### Deliverables
- Daily trading logs and P&L reports
- Agent decision audit trail (full reasoning)
- Performance dashboard with key metrics
- Risk metrics tracking (Sharpe, drawdown, win rate)
- Optimization recommendations report

#### Success Criteria
- Win rate: 48-55% (realistic expectation)
- Sharpe ratio: 1.6-2.0 (with real drawdowns)
- Monthly return: 1.5-3.5% (conservative)
- Maximum drawdown: <-15%
- System uptime: >99%

---

### Phase 3: Enhancement (Q1 2026)
**Target:** 2026-02-16 to 2026-03-31
**Status:** Planning

#### Goals
- [ ] Extend stock coverage to 150-200 stocks
- [ ] Implement additional ML models (LSTM, XGBoost ensemble)
- [ ] Add multi-timeframe analysis (1H, 4H, daily confluence)
- [ ] Develop sector-specific trading rules
- [ ] Create automated hyperparameter optimization
- [ ] Add backtesting optimization engine
- [ ] Implement portfolio rebalancing strategies

#### Deliverables
- Extended model library (150+ stocks)
- Multi-timeframe analysis module
- Sector rules configuration
- Parameter optimization framework
- Enhanced dashboard with sector analytics

#### Success Criteria
- Improved model accuracy (+5-10% over baseline)
- Better risk-adjusted returns
- Reduced correlation between positions
- More diversified trading signals

---

### Phase 4: Live Integration (Q2 2026)
**Target:** 2026-04-01 to 2026-06-30
**Status:** Planning

#### Goals
- [ ] Integrate with SSI broker API (real trading)
- [ ] Implement order routing and fill confirmation
- [ ] Add position monitoring with live P&L
- [ ] Create live trading safeguards and circuit breakers
- [ ] Implement real account fund management
- [ ] Add regulatory compliance logging (regulatory audit trail)
- [ ] Develop incident response procedures

#### Deliverables
- SSI API integration module
- Live trading safeguard system
- Fund management framework
- Regulatory compliance system
- Incident response playbook

#### Success Criteria
- Successful paper→live transition
- 100% order accuracy with broker
- <1s execution latency
- Full compliance with broker/regulatory requirements
- Zero critical incidents in first month

---

### Phase 5: Scale & Optimize (Q3-Q4 2026)
**Target:** 2026-07-01 to 2026-12-31
**Status:** Planning

#### Goals
- [ ] Extend to all 300+ HNX/HOSE stocks
- [ ] Implement portfolio optimization (min variance, max Sharpe)
- [ ] Add international market integration (Singapore, Thailand)
- [ ] Develop mobile app for alerts and position monitoring
- [ ] Create multi-user support (fund management)
- [ ] Implement machine learning model retraining pipeline
- [ ] Add performance attribution analysis

#### Deliverables
- Complete stock market coverage (300+)
- Portfolio optimization engine
- International market adapters
- Mobile app (iOS/Android)
- Multi-user management system
- Model retraining pipeline
- Performance analytics dashboard

#### Success Criteria
- 300+ stocks monitored simultaneously
- Portfolio Sharpe ratio: >2.0
- Mobile app 50K+ downloads
- Support 10+ concurrent users
- Model retraining monthly

---

## Feature Roadmap by Category

### 1. Agent System Enhancements
**Current:** 6 fixed agents (Bull, Bear, Alex, Scout, RiskDoctor, Chief)
**Short-term (Q1):** Configurable agent weights and thresholds
**Long-term:** Dynamic agent creation, specialized sector agents

### 2. Signal Generation
**Current:** Path A (ML 3-min) + Path B (News 24/7)
**Short-term:** Add technical pattern recognition (Head & Shoulders, Cup & Handle)
**Long-term:** Integration with alternative data (options flow, whale movements)

### 3. Risk Management
**Current:** Fixed stop-loss (-5%) and take-profit (+15%)
**Short-term:** Dynamic thresholds based on volatility
**Long-term:** VaR-based position sizing, correlation analysis

### 4. Data & Infrastructure
**Current:** CafeF API + parquet files + RSS feeds
**Short-term:** Add PostgreSQL for production data storage
**Long-term:** Cloud data warehouse (BigQuery), real-time data lake

### 5. ML & Models
**Current:** 102 Stockformer models
**Short-term:** Model ensemble (Stockformer + LSTM + XGBoost)
**Long-term:** Custom architectures per sector, online learning

### 6. User Experience
**Current:** Web dashboard (localhost)
**Short-term:** Cloud deployment, multi-user access
**Long-term:** Mobile app, voice commands, AR visualization

---

## Known Limitations & Technical Debt

### Current Limitations
1. **Paper Trading Only** - No live broker integration yet
2. **Single Model Type** - Only Stockformer (needs ensemble)
3. **Vietnamese-Only News** - No international news feeds
4. **Manual Configuration** - Thresholds need manual tuning
5. **Limited Stock Coverage** - 102 stocks (289 available)
6. **No Multi-User** - Single user system
7. **Localhost Only** - No cloud deployment

### Technical Debt
1. **Agent System** - Could benefit from reinforcement learning for weights
2. **News Processing** - Vietnamese NLP could be improved with custom models
3. **Data Pipeline** - Hardcoded RSS sources, needs abstraction
4. **Dashboard** - React frontend could use state management (Redux)
5. **Testing** - Need more integration tests for full trading cycles
6. **Documentation** - Some modules lack comprehensive docstrings

### Planned Debt Reduction
- **Q1 2026:** Implement configurable agent system, abstract news sources
- **Q2 2026:** Add Redux state management, comprehensive test suite
- **Q3 2026:** Custom Vietnamese NLP models, cloud deployment
- **Q4 2026:** Reinforcement learning agent weights, full documentation

---

## Dependencies & Blockers

### External Dependencies
1. **CafeF API** - Real-time market data (critical)
2. **RSS Feeds** - News sources (critical for Path B)
3. **Parquet Data** - Historical fallback (critical for model training)
4. **SSI Broker** - Live trading integration (Q2 2026)
5. **Cloud Provider** - GCP/AWS (Q3 2026)

### Internal Dependencies
1. **Model Library** - 102 Stockformer models (training completed)
2. **Database** - PostgreSQL setup (optional, for scaling)
3. **Testing** - Full test coverage (85% completed)
4. **Documentation** - Comprehensive guides (in progress)

### Potential Blockers
- CafeF API changes or rate limiting
- Vietnamese NLP library availability
- Cloud provider credential/quota issues
- Broker API documentation gaps
- Regulatory changes in Vietnam market

---

## Success Metrics by Phase

### Phase 2 (Validation)
- Win rate: 48-55%
- Sharpe ratio: 1.6-2.0
- Monthly return: 1.5-3.5%
- System uptime: >99%
- Order accuracy: 100%

### Phase 3 (Enhancement)
- Model accuracy +5-10% vs baseline
- Stock coverage: 150-200
- Win rate: 50-55%
- Sharpe ratio: 1.8-2.1

### Phase 4 (Live)
- Successful paper→live transition
- Order fill rate: >99%
- Execution latency: <1s
- Regulatory compliance: 100%

### Phase 5 (Scale)
- Stock coverage: 300+
- Portfolio Sharpe: >2.0
- User base: 10+ concurrent
- Model retraining: Monthly

---

## Budget & Resource Allocation

### Development Resources
- **Q1 2026:** 2 FTE (dev/ops)
- **Q2 2026:** 3 FTE (add broker integration engineer)
- **Q3 2026:** 4 FTE (add DevOps/cloud engineer)
- **Q4 2026:** 4-5 FTE (add data scientist)

### Infrastructure Costs
- **Phase 1-2:** Minimal ($0-50/month, localhost + CafeF API)
- **Phase 3-4:** Low ($100-300/month, cloud + broker feed)
- **Phase 5:** Moderate ($500-1000/month, scale + data warehouse)

### ML Training/Data
- Historical data: Already downloaded (289 stocks)
- Model training: Once per month (~$20/month compute)
- Alternative data: TBD (Phase 5)

---

## Risk Mitigation

### Market Risks
- **Risk:** Market regime changes make models irrelevant
- **Mitigation:** Regular model retraining, regime detection system

### Operational Risks
- **Risk:** CafeF API downtime
- **Mitigation:** Multiple data source fallback, historical data cache

- **Risk:** Broker API changes
- **Mitigation:** Abstracted broker interface, rapid response team

### Technical Risks
- **Risk:** Memory leaks in long-running system
- **Mitigation:** Profiling, resource monitoring, auto-restart scripts

- **Risk:** Model degradation over time
- **Mitigation:** Continuous accuracy monitoring, automated retraining

### Regulatory Risks
- **Risk:** Vietnam market rules changes
- **Mitigation:** Monitor regulatory sources, quick rule update process

---

## Communication & Stakeholder Updates

### Weekly Updates (Phase 2)
- Trading P&L and key metrics
- Signal quality assessment
- Any issues or blockers
- Plan for next week

### Monthly Reviews (All Phases)
- Executive summary of progress
- Achieved vs planned milestones
- Risk assessment
- Resource requests
- Next month priorities

### Quarterly Planning (All Phases)
- Strategic alignment review
- Budget and resource planning
- Competitive analysis
- Next quarter detailed plan

---

## References

- Backtest Results: See `docs/project-overview-pdr.md` (Section: Success Metrics)
- Current Code: See `docs/codebase-summary.md` for module breakdown
- Deployment: See `docs/deployment-guide.md` for setup instructions
- Architecture: See `docs/system-architecture.md` for detailed design

---

*This roadmap is a living document. Updates will be made weekly during Phase 2 validation, then quarterly. All teams should review this roadmap during their planning cycles.*

**Document Owner:** VN-Quant Development Team
**Next Review:** 2026-03-04 (weekly during Phase 2)
**Last Updated:** 2026-02-25
