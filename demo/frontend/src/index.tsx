import { Provider } from "mobx-react";
import ReactDOM from "react-dom";
import { BrowserRouter } from "react-router-dom";
import RootStore from "stores/rootStore";
import App from "./App";
const rootStore = new RootStore();

ReactDOM.render(
  <Provider {...rootStore}>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </Provider>,
  document.getElementById("root")
);
