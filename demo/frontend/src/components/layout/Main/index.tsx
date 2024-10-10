import { Affix, Drawer, Layout } from "antd";
import React, { ReactNode, useState } from "react";
import { useLocation } from "react-router-dom";
import Footer from "../Footer";
import Header from "../Header";
import Sidenav from "../Sidebar";
import { observer } from "mobx-react";
import { useStores } from "stores/rootStore";
import Loading from "components/Loading";

const { Header: AntHeader, Content, Sider } = Layout;

interface MainProps {
  children: ReactNode;
}

const Main: React.FC<MainProps> = observer(({ children }) => {
  const { functionalityStore } = useStores();

  const [visible, setVisible] = useState<boolean>(false);
  const [sidenavColor, setSidenavColor] = useState<string>("#1890ff");
  const [sidenavType, setSidenavType] = useState<string>("transparent");
  const [fixed, setFixed] = useState<boolean>(false);

  const openDrawer = () => setVisible(!visible);
  const handleSidenavType = (type: string) => setSidenavType(type);
  const handleSidenavColor = (color: string) => setSidenavColor(color);
  const handleFixedNavbar = (type: boolean) => setFixed(type);

  let { pathname } = useLocation();
  pathname = pathname.replace("/", "");

  const [collapsed, setCollapsed] = useState(false);
  return (
    <Layout
      className={`layout-dashboard ${
        pathname === "profile" ? "layout-profile" : ""
      } ${pathname === "rtl" ? "layout-dashboard-rtl" : ""}`}
    >
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={(value) => setCollapsed(value)}
        breakpoint="lg"
        collapsedWidth="0"
        trigger={null}
        width={250}
        theme="light"
        className={`sider-primary ant-layout-sider-primary ${
          sidenavType === "#fff" ? "active-route" : ""
        }`}
        style={{ background: sidenavType }}
      >
        <Sidenav collapsed={false} />
      </Sider>
      <Layout>
        {fixed ? (
          <Affix>
            <AntHeader className={`${fixed ? "ant-header-fixed" : ""}`}>
              <Header
                onPress={openDrawer}
                name={pathname}
                subName={pathname}
                handleSidenavColor={handleSidenavColor}
                handleSidenavType={handleSidenavType}
                handleFixedNavbar={handleFixedNavbar}
              />
            </AntHeader>
          </Affix>
        ) : (
          <AntHeader className={`${fixed ? "ant-header-fixed" : ""}`}>
            <Header
              onPress={openDrawer}
              name={pathname}
              subName={pathname}
              handleSidenavColor={handleSidenavColor}
              handleSidenavType={handleSidenavType}
              handleFixedNavbar={handleFixedNavbar}
            />
          </AntHeader>
        )}
        <Content className="content-ant">{children}</Content>
        <Footer />
      </Layout>
      <Loading loading={functionalityStore.loading} />
    </Layout>
  );
});

export default Main;
