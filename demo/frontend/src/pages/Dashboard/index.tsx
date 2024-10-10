import { Card, Col, Row, Typography } from "antd";
import Loading from "components/Loading";
import EChart from "components/chart/EChart";
import LineChart from "components/chart/LineChart";
import {
  FaAssistiveListeningSystems,
  FaFingerprint,
  FaMusic,
} from "react-icons/fa";
import "./index.css";
import { useDevices } from "pages/Devices/hooks/useDevices";
import { observer } from "mobx-react-lite";
import { useStores } from "stores/rootStore";
import { useSongs } from "pages/Songs/hooks/useSongs";
import { useFingerprints } from "pages/Fingerprints/hooks/useFingerprints";

const Dashboard = observer(() => {
  const { Title } = Typography;
  const { deviceStore, songsStore}  = useStores();
  useDevices();
  useSongs();
  useFingerprints();

  const count = [
    {
      today: "Total Devices",
      title: deviceStore.devices?.length,
      icon: <FaAssistiveListeningSystems />,
      bnb: "bnb2",
    },
    {
      today: "Total Songs",
      title: songsStore.totalSongs,
      icon: <FaMusic />,
      bnb: "bnb2",
    },
    {
      today: "Total Fingerprints",
      title: songsStore.totalFingeprints,
      icon: <FaFingerprint />,
      bnb: "redtext",
    },
  ];

  return (
    <>
      <div className="layout-content">
        <Row className="rowgap-vbox" gutter={[24, 0]}>
          {count.map((c, index) => (
            <Col
              key={index}
              xs={24}
              sm={24}
              md={12}
              lg={6}
              xl={6}
              className="mb-24"
            >
              <Card bordered={false} className="criclebox ">
                <div className="number">
                  <Row align="middle" gutter={[24, 0]}>
                    <Col xs={18}>
                      <span>{c.today}</span>
                      <Title level={3}>
                        {c.title} 
                      </Title>
                    </Col>
                    <Col xs={6}>
                      <div className="icon-box">{c.icon}</div>
                    </Col>
                  </Row>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
        <Loading />
        {/* <Row gutter={[24, 0]}>
          <Col xs={24} sm={24} md={12} lg={12} xl={10} className="mb-24">
            <Card bordered={false} className="criclebox h-full">
              <EChart />
            </Card>
          </Col>
          <Col xs={24} sm={24} md={12} lg={12} xl={14} className="mb-24">
            <Card bordered={false} className="criclebox h-full">
              <LineChart />
            </Card>
          </Col>
        </Row> */}
      </div>
    </>
  );
});

export default Dashboard;
