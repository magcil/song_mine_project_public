import "antd/dist/antd.css";
import { observer } from "mobx-react";
import Dashboard from "pages/Dashboard";
import Demo from "pages/Demo";
import Detected from "pages/Detected";
import Devices from "pages/Devices";
import Songs from "pages/Songs";
import { useEffect } from "react";
import { Route, Switch, useHistory } from "react-router-dom";
import PageNotFound from "routes/PageNotFound";
import ProtectedRoute from "routes/ProtectedRoute";
import authService from "services/authService";
import "./assets/styles/main.css";
import "./assets/styles/responsive.css";
import Main from "./components/layout/Main";
import SignIn from "./pages/SignIn";

const App: React.FC = observer(() => {
  const history = useHistory();
  useEffect(() => {
    if (history.location.pathname === "/") return;

    const token = localStorage.getItem("token");
    if (token) {
      authService.loggedin();
    } else {
      history.push("/sign-in");
    }
  }, []);
  return (
    <div className="App">
      <Switch>
        <Route path="/" exact component={Demo} />
        <Route path="/sign-in" exact component={SignIn} />
        <Main>
          <Switch>
            <ProtectedRoute
              path="/dashboard"
              component={Dashboard}
              isAuthenticated={true}
            />
            <ProtectedRoute
              path="/devices"
              component={Devices}
              isAuthenticated={true}
            />
            <ProtectedRoute
              path="/songs"
              component={Songs}
              isAuthenticated={true}
            />
            <ProtectedRoute
              path="/detections"
              component={Detected}
              isAuthenticated={true}
            />
            <Route component={PageNotFound} />
          </Switch>
        </Main>
      </Switch>
    </div>
  );
});

export default App;
