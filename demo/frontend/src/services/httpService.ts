import axios from "axios";
import Consts from "consts/consts";
const httpService = axios.create({
  baseURL: Consts.API_URL,
  timeout: 10000, // request timeout
  headers: {
    "Content-Type": "multipart/form-data",
  },
});

// Request interceptor
httpService.interceptors.request.use(
  (config: any) => {
    // Do something before request is sent, for example, set your auth token from localStorage
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error: any) => {
    // Do something with request error
    return Promise.reject(error);
  }
);

// Response interceptor
httpService.interceptors.response.use(
  (response: any) => {
    // Any status code that lie within the range of 2xx cause this function to trigger
    return response;
  },
  (error: { response: { status: number } }) => {
    // Any status codes that falls outside the range of 2xx cause this function to trigger
    if (error.response.status === 401) {
      // Handle 401 Unauthorized error - for example, redirect to login page
      // this is a generic example, implement based on your needs
      localStorage.removeItem("token");
      // window.location.href = '/sign-in';
    }

    return Promise.reject(error);
  }
);

export default httpService;
