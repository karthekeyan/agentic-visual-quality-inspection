import axios from 'axios'

const API_BASE_URL = 'http://localhost:8742'

const client = axios.create({ baseURL: API_BASE_URL })

export async function inspectImage(file) {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await client.post('/inspect', formData)
  return data
}

export async function fetchHistory(limit = 5) {
  const { data } = await client.get('/history', { params: { limit } })
  return data
}
