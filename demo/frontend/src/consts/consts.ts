const LOCAL_DEV_API = "http://localhost:8080";
const PROD_API = "http://35.90.18.149/";

const Consts = {
  THRESHOLD: 0.04,
  API_URL:
    window.location.origin.match(/(localhost|127\.0\.0\.1|192\.)/) !== null
      ? process.env.REACT_APP_API
        ? process.env.REACT_APP_API
        : LOCAL_DEV_API
      : PROD_API,
};

export default Consts;
