import { Menu } from "antd";
import museek from "assets/svgs/museek_gray.svg";
import { BiSolidDashboard } from "react-icons/bi";
import { BsFillMicFill } from "react-icons/bs";
import { FaAssistiveListeningSystems, FaDatabase } from "react-icons/fa";
import { PiDetectiveBold } from "react-icons/pi";
import { Link, NavLink } from "react-router-dom";
interface SidenavProps {
  collapsed: boolean;
}

const Sidenav: React.FC<SidenavProps> = ({ collapsed }: SidenavProps) => {
  return (
    <>
      <div className="brand">
        <Link to={"/dashboard"}>
          <img src={museek} alt="" />
        </Link>
      </div>
      <hr />
      <Menu theme="dark" mode="inline">
        <Menu.Item key="1">
          <NavLink to="/dashboard">
            <span className="icon">{<BiSolidDashboard />}</span>
            <span className="label">Dashboard</span>
          </NavLink>
        </Menu.Item>
        <Menu.Item key="2">
          <NavLink to="/devices">
            <span className="icon">{<FaAssistiveListeningSystems />}</span>
            <span className="label">Devices</span>
          </NavLink>
        </Menu.Item>
        <Menu.Item key="3">
          <NavLink to="/songs">
            <span className="icon">{<FaDatabase />}</span>
            <span className="label">Songs</span>
          </NavLink>
        </Menu.Item>
        <Menu.Item key="4">
          <NavLink to="/detections">
            <span className="icon">{<PiDetectiveBold />}</span>
            <span className="label">Detected</span>
          </NavLink>
        </Menu.Item>
        <Menu.Item key="5">
          <NavLink to="/" exact>
            <span className="icon">{<BsFillMicFill />}</span>
            <span className="label">Demo</span>
          </NavLink>
        </Menu.Item>
      </Menu>
    </>
  );
};

export default Sidenav;
