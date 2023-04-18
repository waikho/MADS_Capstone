--
-- PostgreSQL database dump
--

-- Dumped from database version 14.7 (Ubuntu 14.7-0ubuntu0.22.04.1)
-- Dumped by pg_dump version 14.7 (Ubuntu 14.7-0ubuntu0.22.04.1)

-- Started on 2023-04-18 10:53:31 CST

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 323 (class 1259 OID 233587)
-- Name: alpaca_daily_selected; Type: TABLE; Schema: public; Owner: capstone
--

CREATE TABLE public.alpaca_daily_selected (
    tranx_date timestamp with time zone NOT NULL,
    symbol character varying NOT NULL,
    open double precision NOT NULL,
    close double precision NOT NULL,
    low double precision NOT NULL,
    high double precision NOT NULL,
    vol double precision NOT NULL,
    vwap double precision NOT NULL,
    trade_count double precision NOT NULL
);


ALTER TABLE public.alpaca_daily_selected OWNER TO capstone;

--
-- TOC entry 3294 (class 2606 OID 233620)
-- Name: alpaca_daily_selected alpaca_daily_selected_pkey; Type: CONSTRAINT; Schema: public; Owner: capstone
--

ALTER TABLE ONLY public.alpaca_daily_selected
    ADD CONSTRAINT alpaca_daily_selected_pkey PRIMARY KEY (tranx_date, symbol);


-- Completed on 2023-04-18 10:53:32 CST

--
-- PostgreSQL database dump complete
--

