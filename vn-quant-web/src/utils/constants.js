// API URL configuration
export const API_URL = import.meta.env.VITE_API_URL || '/api'

// Utility function for formatting money
export const fmtMoney = (n) => new Intl.NumberFormat('vi-VN').format(n)
