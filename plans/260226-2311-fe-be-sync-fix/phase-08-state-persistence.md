# Phase 08: Add Frontend State Persistence

**Priority:** P2 (MEDIUM)
**Effort:** 0.5h
**Status:** Pending

## Context Links
- Frontend: `vn-quant-web/src/App.jsx` - Current state resets on page reload
- Audit: "Add FE state persistence - localStorage for user preferences, selected symbol, etc."

## Overview
Current behavior: All user preferences (selected symbol, active view, API key) reset on page reload.

**Goal:** Persist user state to localStorage for better UX.

## Key Insights
- Use localStorage for non-sensitive preferences
- Sync state to localStorage on change
- Restore state on mount
- Handle localStorage errors gracefully
- Keep state shape simple (JSON serializable)

## Requirements

### Functional
- Persist selected symbol across page reloads
- Remember last active view
- Store API key (from Phase 01)
- Save user preferences (theme, layout settings)
- Restore state on app mount

### Non-Functional
- Sync to localStorage within 100ms of change
- Handle quota exceeded errors
- Backward compatible (handle missing keys)
- Clear sensitive data on logout/reset

## Architecture

```
┌────────────────────────────────┐
│ React State                    │
│ - activeView                   │
│ - analysisSymbol               │
│ - apiKey                       │
│ - preferences                  │
└──────────┬─────────────────────┘
           │ onChange
           ↓
┌────────────────────────────────┐
│ localStorage Sync              │
│ - vn-quant-active-view         │
│ - vn-quant-analysis-symbol     │
│ - vn-quant-api-key             │
│ - vn-quant-preferences         │
└────────────────────────────────┘
           ↑
           │ onMount
┌──────────┴─────────────────────┐
│ App.jsx useEffect              │
│ - Restore state from storage   │
└────────────────────────────────┘
```

## Related Code Files

**Frontend (Create):**
- `vn-quant-web/src/hooks/use-local-storage.js` - Reusable localStorage hook (~40 lines)
- `vn-quant-web/src/utils/storage.js` - Storage helpers (~30 lines)

**Frontend (Modify):**
- `vn-quant-web/src/App.jsx` - Use localStorage hooks for state
- `vn-quant-web/src/views/analysis-view.jsx` - Persist selected symbol

## Implementation Steps

### Step 1: Create Storage Utility
Create `vn-quant-web/src/utils/storage.js`:

```javascript
const STORAGE_PREFIX = 'vn-quant-'

export const storage = {
  get(key, defaultValue = null) {
    try {
      const item = localStorage.getItem(STORAGE_PREFIX + key)
      return item ? JSON.parse(item) : defaultValue
    } catch (err) {
      console.error(`Failed to read ${key} from localStorage:`, err)
      return defaultValue
    }
  },

  set(key, value) {
    try {
      localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value))
    } catch (err) {
      console.error(`Failed to write ${key} to localStorage:`, err)
      // Handle quota exceeded
      if (err.name === 'QuotaExceededError') {
        console.warn('localStorage quota exceeded, clearing old data...')
        storage.clear()
      }
    }
  },

  remove(key) {
    try {
      localStorage.removeItem(STORAGE_PREFIX + key)
    } catch (err) {
      console.error(`Failed to remove ${key} from localStorage:`, err)
    }
  },

  clear() {
    try {
      Object.keys(localStorage)
        .filter(key => key.startsWith(STORAGE_PREFIX))
        .forEach(key => localStorage.removeItem(key))
    } catch (err) {
      console.error('Failed to clear localStorage:', err)
    }
  }
}
```

### Step 2: Create useLocalStorage Hook
Create `vn-quant-web/src/hooks/use-local-storage.js`:

```javascript
import { useState, useEffect } from 'react'
import { storage } from '../utils/storage'

export function useLocalStorage(key, defaultValue) {
  // Initialize with stored value or default
  const [value, setValue] = useState(() => {
    return storage.get(key, defaultValue)
  })

  // Sync to localStorage whenever value changes
  useEffect(() => {
    storage.set(key, value)
  }, [key, value])

  return [value, setValue]
}
```

### Step 3: Update App.jsx to Use Persistence
Modify `vn-quant-web/src/App.jsx`:

```javascript
import { useLocalStorage } from './hooks/use-local-storage'

function App() {
  // BEFORE:
  // const [activeView, setActiveView] = useState('dashboard')
  // const [apiKey, setApiKey] = useState(localStorage.getItem('vn-quant-api-key') || null)

  // AFTER:
  const [activeView, setActiveView] = useLocalStorage('active-view', 'dashboard')
  const [apiKey, setApiKey] = useLocalStorage('api-key', null)
  const [preferences, setPreferences] = useLocalStorage('preferences', {
    theme: 'dark',
    autoRefresh: true,
    refreshInterval: 5000,
  })

  // Rest of component stays the same
  return (
    <div className="flex w-full h-screen bg-[#0a0e17] text-slate-200">
      {/* ... */}
    </div>
  )
}
```

### Step 4: Persist Analysis Symbol
Modify `vn-quant-web/src/views/analysis-view.jsx`:

```javascript
import { useLocalStorage } from '../hooks/use-local-storage'

export function AnalysisView() {
  // BEFORE:
  // const [symbol, setSymbol] = useState('MWG')

  // AFTER:
  const [symbol, setSymbol] = useLocalStorage('analysis-symbol', 'MWG')

  // Rest of component stays the same
  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      {/* Symbol selector */}
      <input
        value={symbol}
        onChange={(e) => setSymbol(e.target.value.toUpperCase())}
        placeholder="Enter symbol..."
      />
      {/* ... */}
    </div>
  )
}
```

### Step 5: Add Preferences Panel (Optional)
Create `vn-quant-web/src/components/preferences-panel.jsx`:

```javascript
import { useLocalStorage } from '../hooks/use-local-storage'

export function PreferencesPanel({ onClose }) {
  const [preferences, setPreferences] = useLocalStorage('preferences', {
    theme: 'dark',
    autoRefresh: true,
    refreshInterval: 5000,
  })

  const updatePref = (key, value) => {
    setPreferences(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="glass-panel rounded-xl max-w-md w-full p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-white">Preferences</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        <div className="space-y-4">
          {/* Auto Refresh */}
          <div className="flex justify-between items-center">
            <label className="text-slate-300">Auto Refresh</label>
            <input
              type="checkbox"
              checked={preferences.autoRefresh}
              onChange={(e) => updatePref('autoRefresh', e.target.checked)}
              className="w-5 h-5"
            />
          </div>

          {/* Refresh Interval */}
          <div>
            <label className="text-slate-300 block mb-2">Refresh Interval (ms)</label>
            <input
              type="number"
              value={preferences.refreshInterval}
              onChange={(e) => updatePref('refreshInterval', parseInt(e.target.value))}
              className="w-full p-2 bg-black/30 border border-white/10 rounded-lg text-white"
              min="1000"
              max="60000"
              step="1000"
            />
          </div>

          {/* Clear Data Button */}
          <button
            onClick={() => {
              if (confirm('Clear all saved preferences?')) {
                localStorage.clear()
                window.location.reload()
              }
            }}
            className="w-full px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
          >
            Clear All Data
          </button>
        </div>
      </div>
    </div>
  )
}
```

### Step 6: Add Preferences to Sidebar
Modify `vn-quant-web/src/components/sidebar.jsx`:

```javascript
import { useState } from 'react'
import { PreferencesPanel } from './preferences-panel'

export function Sidebar({ activeView, setView }) {
  const [showPreferences, setShowPreferences] = useState(false)

  return (
    <aside className="hidden md:flex flex-col w-64 border-r border-white/5">
      {/* ... existing nav ... */}

      <div className="p-4 border-t border-white/5">
        <button
          onClick={() => setShowPreferences(true)}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-slate-400 hover:bg-white/5 hover:text-white transition-all"
        >
          <span className="material-symbols-outlined text-[20px]">settings</span>
          <span className="font-medium">Preferences</span>
        </button>
      </div>

      {showPreferences && <PreferencesPanel onClose={() => setShowPreferences(false)} />}
    </aside>
  )
}
```

### Step 7: Test Persistence
1. Open app, select Analysis view
2. Change symbol to "VNM"
3. Reload page → verify VNM still selected
4. Switch to Trading view
5. Reload page → verify Trading view active
6. Open Preferences, toggle auto refresh
7. Reload page → verify preference saved

## Todo List
- [ ] Create utils/storage.js with get/set/clear helpers
- [ ] Create hooks/use-local-storage.js hook
- [ ] Update App.jsx to persist activeView
- [ ] Update App.jsx to persist apiKey
- [ ] Add preferences state to App.jsx
- [ ] Update AnalysisView to persist symbol
- [ ] Create PreferencesPanel component (optional)
- [ ] Add preferences button to Sidebar (optional)
- [ ] Test: activeView persists on reload
- [ ] Test: symbol persists on reload
- [ ] Test: apiKey persists on reload
- [ ] Test: quota exceeded handling
- [ ] Test: clear data button works

## Success Criteria
- [ ] Active view persists across page reloads
- [ ] Selected analysis symbol persists
- [ ] API key persists (from Phase 01)
- [ ] Preferences saved to localStorage
- [ ] No console errors on storage operations
- [ ] Quota exceeded handled gracefully
- [ ] Clear data button removes all stored data

## Risk Assessment
- **Risk:** localStorage full → Mitigated: Handle QuotaExceededError, clear old data
- **Risk:** Corrupt JSON in storage → Mitigated: Try/catch with fallback to defaults
- **Risk:** Sensitive data leaked → Mitigated: Only store non-sensitive preferences

## Security Considerations
- Don't store sensitive data (passwords, tokens) in localStorage
- API key acceptable (paper trading only)
- Clear localStorage on logout/session end
- Consider encryption for production (future enhancement)

## Next Steps
After completing this phase:
1. Update plan.md with completion status to 100%
2. Run full regression test on all 8 phases
3. Create summary report in plans/reports/
4. Update project documentation with new features
5. Consider Phase 09+ enhancements:
   - TypeScript migration
   - Unit tests for components
   - E2E tests with Playwright
   - Performance optimization
   - Accessibility improvements
